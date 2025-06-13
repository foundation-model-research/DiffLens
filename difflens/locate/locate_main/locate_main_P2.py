import os
from fnmatch import fnmatchcase
from natsort import natsorted
from safetensors.torch import load_model
import torch
from torch import nn, Tensor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import resolve_widths_P2
from .sae import Sae, SaeConfig

# Classification Model
class HiddenLinear(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for i in range(49)])
    
    def forward(self, x, t):
        x = x.reshape(x.shape[0],-1)
        x = self.linears[t](x)
        return x
    
def make_model(path, input_dim, output_dim):
    model = HiddenLinear(input_dim, output_dim).cuda()
    model.load_state_dict(torch.load(path, map_location='cuda:0'))
    model.eval()

    return model

# Integrated Gradients Helper Functions
def scaled_input(attention_probs, batch_size, num_batch):
    # attention_probs: (head_num, h, w)
    baseline = torch.zeros_like(attention_probs)  # (head_num, h, w)

    num_points = batch_size * num_batch
    step = (attention_probs - baseline) / num_points  # (head_num, h, w)

    res = torch.cat([torch.add(baseline, step * i).unsqueeze(0) for i in range(num_points)], dim=0)  # (num_points, head_num, h, w)
    return res, step

# Helper Functions
def get_model_device(model):
    return next(model.parameters()).device

class P2LocateFeature:
    def __init__(
        self, cfg, dataset, model, diffusion,
    ):
        if cfg.hookpoints:
            assert not cfg.layers, "Cannot specify both `hookpoints` and `layers`."

            # Replace wildcard patterns
            raw_hookpoints = []
            for name, _ in model.named_modules():
                if any(fnmatchcase(name, pat) for pat in cfg.hookpoints):
                    raw_hookpoints.append(name)

            # Natural sort to impose a consistent order
            cfg.hookpoints = natsorted(raw_hookpoints)
        else:
            raise NotImplementedError

        self.cfg = cfg
        
        self.distribute_modules()

        N = len(cfg.hookpoints)

        device = get_model_device(model)
        input_widths, h_classification_widths = resolve_widths(model, cfg.hookpoints)
        unique_widths = set(input_widths.values())

        if cfg.distribute_modules and len(unique_widths) > 1:
            # dist.all_to_all requires tensors to have the same shape across ranks
            raise ValueError(
                f"All modules must output tensors of the same shape when using "
                f"`distribute_modules=True`, got {unique_widths}"
            )

        self.model = model
        self.diffusion = diffusion
        self.dataset = dataset
        
        sae_cfg = cfg.sae

        self.saes = {
            hook: Sae(
                    d_in = 512,
                    cfg = sae_cfg,
                    device = device,
                    kernel_size=cfg.kernel_size,
                    )
            for hook in self.local_hookpoints()
        }

        for name, sae in self.saes.items():
            # update model path
            path = cfg.sparse_autoencoder_model_path
            load_model(sae, f"{path}/sae.safetensors", device="cuda")
            sae.eval()
        
        # classification model
        if self.cfg.classifier_name == "gender":
            output_dim=2
        elif self.cfg.classifier_name=="age":
            output_dim=3
        elif self.cfg.classifier_name=="race":
            output_dim=4
        else:
            raise NotImplementedError

        classification_model_path = self.cfg.classification_model_path
        self.h_classification = make_model(classification_model_path, h_classification_widths[self.local_hookpoints()[0]], output_dim=output_dim)
        self.h_classification = self.h_classification

        self.calculation = [0.0 for _ in range(output_dim)]

    def local_hookpoints(self) -> list[str]:
        return (
            self.module_plan[dist.get_rank()]
            if self.module_plan
            else self.cfg.hookpoints
        )

    def distribute_modules(self):
        """Prepare a plan for distributing modules across ranks."""
        if not self.cfg.distribute_modules:
            self.module_plan = []
            print(f"Training on modules: {self.cfg.hookpoints}")
            return

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

    def locate(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_modules

        num_batches = len(self.dataset) // self.cfg.batch_size

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
            desc="Locating", 
            disable=not rank_zero, 
            total=num_batches,
        )
        
        hidden_dict: dict[str, Tensor] = {}
        name_to_module = {
            name: self.model.get_submodule(name) for name in self.cfg.hookpoints
        }
        maybe_wrapped = {}
        module_to_name = {v: k for k, v in name_to_module.items()}

        def hook(module: nn.Module, _, outputs):
            # Maybe unpack tuple outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            name = module_to_name[module]
            
            outputs = outputs.contiguous()
            hidden_dict[name] = outputs

        for num, batch in enumerate(dl):
            hidden_dict.clear()

            # Forward pass on the model to get the next batch of activations            
            handles = [
                mod.register_forward_hook(hook) for mod in name_to_module.values()
            ]
            try:
                with torch.no_grad():
                    indices = list(range(self.cfg.num_timesteps))[::-1]
                    timestep = indices[self.cfg.locate_timestep]
                    timesteps = torch.tensor([timestep] * batch.shape[0], device=device)
                    self.diffusion.ddim_sample(self.model, batch.to(device), timesteps)
            finally:
                for handle in handles:
                    handle.remove()

            if self.cfg.distribute_modules:
                hidden_dict = self.scatter_hiddens(hidden_dict)

            for name, hiddens in hidden_dict.items():
                raw = self.saes[name]  # 'raw' never has a DDP wrapper

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
                
                with torch.no_grad():
                    if self.cfg.classifier_name=="gender":
                        temperature = 8.
                    else:
                        temperature = 1.
                    
                    # Don't use the first block
                    reversed_timestep = 50 - self.cfg.locate_timestep - 1
                    target_attribute = self.h_classification(hiddens, reversed_timestep)/temperature
                    
                    target_attribute = torch.argmax(target_attribute, dim=1)

                wrapped = maybe_wrapped[name]
                pre_acts = wrapped.pre_acts(hiddens)
                top_acts, top_indices = wrapped.select_topk(pre_acts)
                batch_size, k, height, width = top_acts.shape

                sparse_latents = torch.zeros((batch_size, wrapped.num_latents, height, width), 
                                     device=wrapped.device, dtype=wrapped.dtype)
                
                sparse_latents.scatter_(1, top_indices, top_acts)

                scaled_weights, weights_step = scaled_input(sparse_latents.detach(), batch_size = self.cfg.scale_batch_size, num_batch = self.cfg.num_batch)
                current_time_grad = None       
                   
                for batch_idx in range(self.cfg.num_batch):
                    batch_weights = scaled_weights[batch_idx * self.cfg.scale_batch_size:(batch_idx + 1) * self.cfg.scale_batch_size].squeeze(0)
                    batch_weights.requires_grad = True
                    batch_weights.retain_grad()
                    if self.cfg.classifier_name == "gender":
                        temperature = 8.
                    elif self.cfg.classifier_name == "age":
                        temperature = 1.
                    elif self.cfg.classifier_name == "race":
                        temperature = 1.
                    
                    recons = wrapped.decoder(batch_weights)
                    
                    logits = self.h_classification(recons, reversed_timestep)/temperature
                    logits = torch.nn.functional.softmax(logits, dim=1)
                    # print(logits)
                    
                    logits[:, target_attribute].backward()
                    
                    grad = batch_weights.grad
                    current_time_grad = current_time_grad+grad if current_time_grad is not None else grad
                
                integrated_gradients = (current_time_grad*weights_step)
                results = torch.gather(integrated_gradients, 1, top_indices).cpu().to(torch.float16)

                if isinstance(self.calculation[0], float):
                    self.calculation = [torch.zeros_like(integrated_gradients) for _ in self.calculation ]
                
                self.calculation[target_attribute] +=  integrated_gradients

                if num%200 == 1:
                    # Save for each 200 iters.
                    for attr, calculation in enumerate(self.calculation):
                        save_path = os.path.join(self.cfg.save_path,
                                                f"kernel_{self.cfg.kernel_size}", 
                                                f"k_{self.cfg.ae_k}", 
                                                f"expansion_factor_{self.cfg.ae_expansion_factor}", 
                                                f"{self.cfg.classifier_name}",
                                                f"timestep_{self.cfg.locate_timestep}",
                                                f"target_{int(attr)}")
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(calculation.detach().cpu() / num, os.path.join(save_path, f"calculation.pt"))

            pbar.update()

        pbar.close()