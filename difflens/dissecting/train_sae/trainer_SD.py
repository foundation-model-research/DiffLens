import os
from collections import defaultdict
from dataclasses import asdict
from typing import Sized
import copy

import torch
import torch.distributed as dist
from datasets import Dataset as HfDataset
from fnmatch import fnmatchcase
from natsort import natsorted
from safetensors.torch import load_model
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from .config import TrainConfig
from .data import MemmapDataset, PromptDataset
from .sae import Sae
from .utils import geometric_median, get_layer_list, resolve_widths_SD

from torch.utils.tensorboard import SummaryWriter

from diffusers import StableDiffusionPipeline

# helper functions
def get_model_device(model):
    return next(model.unet.parameters()).device

def random_select(outputs, num_samples=256):
    """
    Randomly selects `num_samples` rows from the input tensor `outputs`.

    Args:
        outputs (torch.Tensor): Input tensor of shape [n, d].
        num_samples (int): Number of samples to select.

    Returns:
        torch.Tensor: A tensor of shape [num_samples, d] containing randomly selected rows.
    """
    n = outputs.shape[0]
    
    indices = torch.randperm(n)[:num_samples]
    
    selected_outputs = outputs[indices]
    
    return selected_outputs

class SDSaeTrainer:
    def __init__(
        self, cfg: TrainConfig, dataset: PromptDataset, model: StableDiffusionPipeline,
    ):
        if cfg.hookpoints:
            assert not cfg.layers, "Cannot specify both `hookpoints` and `layers`."

            # Replace wildcard patterns
            raw_hookpoints = []
            for name, _ in model.unet.named_modules():
                # if any(fnmatchcase(name, pat) for pat in cfg.hookpoints):
                #     raw_hookpoints.append(name)
                
                # We only use mid_block
                if any(fnmatchcase(name, pat) for pat in ['mid_block']):
                    raw_hookpoints.append(name)

            # Natural sort to impose a consistent order
            cfg.hookpoints = natsorted(raw_hookpoints)
        else:
            raise NotImplementedError

        self.cfg = cfg
        
        # Split the dataset into train and eval
        train_samples = len(dataset)
        
        self.dataset = dataset.select(range(train_samples))
        
        print(f"Training samples: {train_samples}")

        # We only train on mid_block
        self.cfg.hookpoints = [0]
        self.distribute_modules()

        num_examples = len(dataset)

        device = get_model_device(model)
        input_widths = resolve_widths_SD(model, ["mid_block"])
        unique_widths = set(input_widths.values())

        if cfg.distribute_modules and len(unique_widths) > 1:
            # dist.all_to_all requires tensors to have the same shape across ranks
            raise ValueError(
                f"All modules must output tensors of the same shape when using "
                f"`distribute_modules=True`, got {unique_widths}"
            )

        self.model = model

        # We only train for bottleneck
        bottleneck_name = 'mid_block'

        if cfg.distribute_modules and len(unique_widths) > 1:
            # dist.all_to_all requires tensors to have the same shape across ranks
            raise ValueError(
                f"All modules must output tensors of the same shape when using "
                f"`distribute_modules=True`, got {unique_widths}"
            )
        
        self.saes = {
            hook: Sae(input_widths[bottleneck_name], cfg.sae, device, kernel_size = cfg.kernel_size)
            for hook in self.local_hookpoints()
        }

        pgs = [
            {
                "params": sae.parameters(),
                # Auto-select LR using 1 / sqrt(d) scaling law from Fig 3 of the paper
                "lr": cfg.lr or 2e-4 / (sae.num_latents / (2**14)) ** 0.5
            }
            for sae in self.saes.values()
        ]
        # Dedup the learning rates we're using, sort them, round to 2 decimal places
        lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]
        print(f"Learning rates: {lrs}" if len(lrs) > 1 else f"Learning rate: {lrs[0]}")

        try:
            from bitsandbytes.optim import Adam8bit as Adam

            print("Using 8-bit Adam from bitsandbytes")
        except ImportError:
            from torch.optim import Adam

            print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
            print("Run `pip install bitsandbytes` for less memory usage.")

        self.global_step = 0
        self.num_tokens_since_fired = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.long)
            for name, sae in self.saes.items()
        }
        self.optimizer = Adam(pgs)

        total_steps = num_examples // cfg.batch_size // 2
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.lr_warmup_ratio*total_steps, num_examples // cfg.batch_size // 2
        )

    def load_state(self, path: str):
        """Load the trainer state from disk."""
        device = get_model_device(self.model)

        path = os.path.join(path, f'k_{self.cfg.sae.k}/expansion_factor_{self.cfg.sae.expansion_factor}/lr_{self.cfg.lr}')

        # Load the train state first so we can print the step number
        train_state = torch.load(f"{path}/state.pt", map_location=device, weights_only=True)
        self.global_step = train_state["global_step"]
        self.num_tokens_since_fired = train_state["num_tokens_since_fired"]

        print(f"\033[92mResuming training at step {self.global_step} from '{path}'\033[0m")

        lr_state = torch.load(f"{path}/lr_scheduler.pt", map_location=device, weights_only=True)
        opt_state = torch.load(f"{path}/optimizer.pt", map_location=device, weights_only=True)
        self.optimizer.load_state_dict(opt_state)
        self.lr_scheduler.load_state_dict(lr_state)

        for name, sae in self.saes.items():
            print("load_model name: ", name)
            load_model(sae, f"{path}/{name}/sae.safetensors", device=str(device))

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_modules

        self.cfg.log_to_wandb = True

        if self.cfg.log_to_wandb and rank_zero:
            # Replace Tensorboard
            try:
                log_dir = f"runs/"
                self.writer = SummaryWriter(log_dir=log_dir)
                print(f"TensorBoard logs will be saved to {log_dir}")
            except ImportError:
                print("TensorBoard not installed, skipping logging.")
                self.cfg.log_to_tensorboard = False   

        num_sae_params = sum(
            p.numel() for s in self.saes.values() for p in s.parameters()
        )
        num_model_params = sum(p.numel() for p in self.model.unet.parameters())
        print(f"Number of SAE parameters: {num_sae_params:_}")
        print(f"Number of model parameters: {num_model_params:_}")

        num_batches = len(self.dataset) // self.cfg.batch_size
        if self.global_step > 0:
            assert hasattr(self.dataset, "select"), "Dataset must implement `select`"

            n = self.global_step * self.cfg.batch_size
            ds = self.dataset.select(range(n, len(self.dataset)))  # type: ignore
        else:
            ds = self.dataset

        device = get_model_device(self.model)
        dl = DataLoader(
            ds, # type: ignore
            batch_size=self.cfg.macro_batch_size,
            # NOTE: We do not shuffle here for reproducibility; the dataset should
            # be shuffled before passing it to the trainer.
            shuffle=False,
        )
        pbar = tqdm(
            desc="Training", 
            disable=not rank_zero, 
            initial=self.global_step, 
            total=num_batches,
        )

        did_fire = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
            for name, sae in self.saes.items()
        }
        num_tokens_in_step = 0

        # For logging purposes
        avg_auxk_loss = defaultdict(float)
        avg_fvu = defaultdict(float)
        avg_multi_topk_fvu = defaultdict(float)

        hidden_dict: dict[str, Tensor] = {"mid_block": []}

        name_to_module = {
            name: self.model.unet.get_submodule(name) for name in ['mid_block'] # We only train on bottleneck
        }
        maybe_wrapped = {}
        module_to_name = {v: k for k, v in name_to_module.items()}

        all_timesteps = list(range(0, 51))
        all_timesteps = list(reversed(all_timesteps))

        hidden_all_timesteps = copy.deepcopy(all_timesteps)

        def hook(module: nn.Module, _, outputs):
            # name = module_to_name[module]            
            name = hidden_all_timesteps.pop()
            hidden_dict[name] = outputs

            # if 'mid_block' in hidden_dict.keys():
            #     hidden_dict[name].append(outputs)
            # else:
            #     hidden_dict[name] = [outputs]

        for batch in dl:
            hidden_dict.clear()
            hidden_all_timesteps = copy.deepcopy(all_timesteps)

            # Bookkeeping for dead feature detection
            # num_tokens_in_step += batch.shape[0]
            num_tokens_in_step += len(batch)

            # Forward pass on the model to get the next batch of activations            
            handles = [
                mod.register_forward_hook(hook) for mod in name_to_module.values()
            ]

            try:
                with torch.no_grad():
                    self.model(batch, num_inference_steps=50)

            finally:
                for handle in handles:
                    handle.remove()
            
            if self.cfg.distribute_modules:
                hidden_dict = self.scatter_hiddens(hidden_dict)
            
            # keys_to_delete = [k for k in hidden_dict if k not in self.cfg.hookpoints]
            keys_to_delete = [k for k in hidden_dict if k not in list(range(*self.cfg.train_timestep_range))]
            for k in keys_to_delete:
                del hidden_dict[k]

            for name, hiddens in hidden_dict.items():
                raw = self.saes[0]  # 'raw' never has a DDP wrapper

                # On the first iteration, initialize the decoder bias
                # if self.global_step == 0:
                    # NOTE: The all-cat here could conceivably cause an OOM in some
                    # cases, but it's unlikely to be a problem with small world sizes.
                    # We could avoid this by "approximating" the geometric median
                    # across all ranks with the mean (median?) of the geometric medians
                    # on each rank. Not clear if that would hurt performance.
                    # median = geometric_median(self.maybe_all_cat(hiddens))
                    # raw.decoder.bias.data = median.to(raw.dtype)

                if not maybe_wrapped:
                    # Wrap the SAEs with Distributed Data Parallel. We have to do this
                    # after we set the decoder bias, otherwise DDP will not register
                    # gradients flowing to the bias after the first step.
                    maybe_wrapped = (
                        {
                            name: DDP(sae, device_ids=[dist.get_rank()])
                            for name, sae in self.saes.items()
                        }
                        if ddp
                        else self.saes
                    )

                # Make sure the W_dec is still unit-norm
                if raw.cfg.normalize_decoder:
                    raw.set_decoder_norm_to_unit_norm()

                acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
                denom = acc_steps * self.cfg.wandb_log_frequency
                wrapped = maybe_wrapped[0] # We only use one SAE for all timesteps

                # Save memory by chunking the activations
                for chunk in hiddens.chunk(self.cfg.micro_acc_steps):
                    out = wrapped(
                        chunk,
                        dead_mask=(
                            # self.num_tokens_since_fired[name]
                            # > self.cfg.dead_feature_threshold
                            # if self.cfg.auxk_alpha > 0
                            # else None
                            None
                        ),
                    )

                    avg_fvu[name] += float(
                        self.maybe_all_reduce(out.fvu.detach()) / denom
                    )
                    if self.cfg.auxk_alpha > 0:
                        avg_auxk_loss[name] += float(
                            self.maybe_all_reduce(out.auxk_loss.detach()) / denom
                        )
                    if self.cfg.sae.multi_topk:
                        avg_multi_topk_fvu[name] += float(
                            self.maybe_all_reduce(out.multi_topk_fvu.detach()) / denom
                        )

                    loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss + out.multi_topk_fvu / 8
                    loss.div(acc_steps).backward()

                    # Update the did_fire mask
                    # Sorry, we don't use the dead feature mask
                    # did_fire[name][out.latent_indices.flatten()] = True
                    # self.maybe_all_reduce(did_fire[name], "max")  # max is boolean "any"

                # Clip gradient norm independently for each SAE
                torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)

            # Check if we need to actually do a training step
            step, substep = divmod(self.global_step + 1, self.cfg.grad_acc_steps)
            if substep == 0:
                # for name, sae in self.saes.items():
                #     print(name, sae.W_dec.data.mean())
                #     if sae.W_dec.grad is None:
                #         print(f"SAE {name} has no gradient")
                # if self.cfg.sae.normalize_decoder:
                #     for sae in self.saes.values():
                #         sae.remove_gradient_parallel_to_decoder_directions()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                ###############
                with torch.no_grad():
                    # Sorry, we don't use the dead feature mask
                    pass
                    # Update the dead feature mask
                    # for name, counts in self.num_tokens_since_fired.items():
                    #     counts += num_tokens_in_step
                    #     counts[did_fire[name]] = 0

                    # Reset stats for this step
                    # num_tokens_in_step = 0
                    # for mask in did_fire.values():
                    #     mask.zero_()

                if (
                    self.cfg.log_to_wandb
                    and (step + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    info = {}

                    for name in hidden_dict:
                        # Sorry, we don't use the dead feature mask
                        # mask = (
                        #     self.num_tokens_since_fired[name]
                        #     > self.cfg.dead_feature_threshold
                        # )

                        info.update(
                            {
                                f"fvu/{name}": avg_fvu[name],
                                # Sorry, we don't use the dead feature mask
                                # f"dead_pct/{name}": mask.mean(
                                #     dtype=torch.float32
                                # ).item(),
                            }
                        )
                        if self.cfg.auxk_alpha > 0:
                            info[f"auxk/{name}"] = avg_auxk_loss[name]
                        if self.cfg.sae.multi_topk:
                            info[f"multi_topk_fvu/{name}"] = avg_multi_topk_fvu[name]

                    avg_auxk_loss.clear()
                    avg_fvu.clear()
                    avg_multi_topk_fvu.clear()

                    if self.cfg.distribute_modules:
                        outputs = [{} for _ in range(dist.get_world_size())]
                        dist.gather_object(info, outputs if rank_zero else None)
                        info.update({k: v for out in outputs for k, v in out.items()})
                    
                    if rank_zero:
                        for key, value in info.items():
                            self.writer.add_scalar(key, value, step)
                            if step%10 == 0:
                                print(f"{name}, {key}: ", value)

                if (step + 1) % self.cfg.save_every == 0:
                    self.save()
                
            self.global_step += 1
            pbar.update()

        self.save()
        pbar.close()

    def local_hookpoints(self) -> list[str]:
        return (
            self.module_plan[dist.get_rank()]
            if self.module_plan
            else self.cfg.hookpoints
        )

    def maybe_all_cat(self, x: Tensor) -> Tensor:
        """Concatenate a tensor across all processes."""
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")

        return x

    def distribute_modules(self):
        """Prepare a plan for distributing modules across ranks."""
        if not self.cfg.distribute_modules:
            self.module_plan = []
            print(f"Training on modules: {self.cfg.hookpoints}")
            return
        
        print("len(self.cfg.hookpoints): ", self.cfg.hookpoints)
        layers_per_rank, rem = divmod(len(self.cfg.hookpoints), dist.get_world_size())
        assert rem == 0, "Number of modules must be divisible by world size"

        # Each rank gets a subset of the layers
        self.module_plan = [
            self.cfg.hookpoints[start : start + layers_per_rank]
            for start in range(0, len(self.cfg.hookpoints), layers_per_rank)
        ]
        for rank, modules in enumerate(self.module_plan):
            print(f"Rank {rank} modules: {modules}")

    def scatter_hiddens(self, hidden_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Scatter & gather the hidden states across ranks."""
        outputs = [
            # Add a new leading "layer" dimension to each tensor
            torch.stack([hidden_dict[hook] for hook in hookpoints], dim=1)
            for hookpoints in self.module_plan
        ]
        local_hooks = self.module_plan[dist.get_rank()]
        shape = next(iter(hidden_dict.values())).shape

        # Allocate one contiguous buffer to minimize memcpys
        buffer = outputs[0].new_empty(
            # The (micro)batch size times the world size
            shape[0] * dist.get_world_size(),
            # The number of layers we expect to receive
            len(local_hooks),
            # All other dimensions
            *shape[1:],
        )

        # Perform the all-to-all scatter
        inputs = buffer.split([len(output) for output in outputs])
        dist.all_to_all([x for x in inputs], outputs)

        # Return a list of results, one for each layer
        return {hook: buffer[:, i] for i, hook in enumerate(local_hooks)}

    def save(self):
        """Save the SAEs to disk."""

        path = self.cfg.run_name or f"./sparse_autoencoder_conv/sae-ckpts-large-kernel/kernel_{self.cfg.kernel_size}/k_{self.cfg.sae.k}/expansion_factor_{self.cfg.sae.expansion_factor}/lr_{self.cfg.lr}"
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        if rank_zero or self.cfg.distribute_modules:
            print("Saving checkpoint")

            for hook, sae in self.saes.items():
                assert isinstance(sae, Sae)

                sae.save_to_disk(f"{path}/{hook}_step_{self.global_step}")
                sae.save_to_disk(f"{path}/{hook}")
    
        if rank_zero:
            torch.save(self.lr_scheduler.state_dict(), f"{path}/lr_scheduler.pt")
            torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
            torch.save({
                "global_step": self.global_step,
                "num_tokens_since_fired": self.num_tokens_since_fired,
            }, f"{path}/state.pt")

            self.cfg.save_json(f"{path}/config.json")

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()

