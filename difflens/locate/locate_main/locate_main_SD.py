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
import numpy as np

from .utils import resolve_widths_SD
from .sae import Sae, SaeConfig

import random

occupations = [
    "firefighter",
    "doctor",
    "receptionist",
    "nurse"
]

# Classification Model
class HiddenLinear(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for i in range(48)])
    
    def forward(self, x, t):
        # x = x.reshape(x.shape[0]//2,-1)
        x = x.reshape(x.shape[0],-1)
        x = self.linears[t](x)
        return x

def make_model(path, input_dim, output_dim):
    model = HiddenLinear(input_dim, output_dim).cuda().to(torch.float16)
    model.load_state_dict(torch.load(path, map_location='cuda:0'))
    model.eval()

    return model

# Integrated Gradients Helper Functions
def scaled_input(attention_probs, batch_size, num_batch):
    baseline = torch.zeros_like(attention_probs)

    num_points = batch_size * num_batch
    step = (attention_probs - baseline) / num_points

    res = torch.cat([torch.add(baseline, step * i).unsqueeze(0) for i in range(num_points)], dim=0)  # (num_points, head_num, h, w)
    return res, step

def scaled_input_pair_baseline(attention_probs, batch_size, num_batch):
    # We replace baseline with negative attribute
    bottleneck_uncond, bottleneck_text = attention_probs.chunk(2, dim=0)
    baseline = torch.cat([bottleneck_uncond[1].unsqueeze(0), bottleneck_text[1].unsqueeze(0)], dim=0)
    attention_probs = torch.cat([bottleneck_uncond[0].unsqueeze(0), bottleneck_text[0].unsqueeze(0)], dim=0)
    
    # We just use the conditional part
    baseline = baseline[1:]
    attention_probs = attention_probs[1:]

    num_points = batch_size * num_batch
    step = (attention_probs - baseline) / num_points  # (head_num, h, w)

    res = torch.cat([torch.add(baseline, step * i).unsqueeze(0) for i in range(num_points)], dim=0)  # (num_points, head_num, h, w)
    return res, step

# Helper Functions
def get_model_device(model):
    return next(model.parameters()).device

class SDLocateFeature:
    def __init__(
        self, cfg, dataset, model, 
    ):
        if cfg.hookpoints:
            assert not cfg.layers, "Cannot specify both `hookpoints` and `layers`."

            # Replace wildcard patterns
            raw_hookpoints = []
            for name, _ in model.unet.named_modules():
                if any(fnmatchcase(name, pat) for pat in cfg.hookpoints):
                    raw_hookpoints.append(name)

            # Natural sort to impose a consistent order
            cfg.hookpoints = natsorted(raw_hookpoints)
        else:
            raise NotImplementedError

        self.cfg = cfg
        
        self.distribute_modules()

        N = len(cfg.hookpoints)
        num_examples = len(dataset)

        device = get_model_device(model.unet)
        input_widths, h_classification_widths = resolve_widths_SD(model, cfg.hookpoints)
        unique_widths = set(input_widths.values())

        if cfg.distribute_modules and len(unique_widths) > 1:
            # dist.all_to_all requires tensors to have the same shape across ranks
            raise ValueError(
                f"All modules must output tensors of the same shape when using "
                f"`distribute_modules=True`, got {unique_widths}"
            )

        self.model = model
        self.dataset = dataset
        
        sae_cfg = SaeConfig()
        # update sae config
        sae_cfg.k = cfg.ae_k
        sae_cfg.expansion_factor = cfg.ae_expansion_factor
        sae_cfg.lr = cfg.ae_lr

        self.saes = {
            hook: {
                0: Sae(
                    d_in = 1280,
                    cfg = sae_cfg,
                    device = device,
                    kernel_size = cfg.kernel_size,
                    ).to(torch.float16)
            }
            for hook in self.local_hookpoints()
        }

        for name, sae in self.saes.items():
            # update model path
            path = cfg.sparse_autoencoder_model_path
            load_model(sae[0], f"{path}/sae.safetensors", device="cuda")
            sae[0].eval()
        
        # Define prompt
        gender_prompts = [
            "A photo of a male person", "A photo of a female person"
        ]

        race_prompts = [
            "A photo of a White person", "A photo of a Black person",
            "A photo of an Asian person", "A photo of an Indian person"
        ]

        age_prompts = [
            "A photo of a child person", "A photo of an adult person", "A photo of an old person"
        ]
        
        # classification model
        if self.cfg.classifier_name == "gender":
            # self.prompt_list = ["A face photo of of a male people", "A face photo of of a female people"]
            self.prompt_list = gender_prompts
            output_dim=2
        elif self.cfg.classifier_name=="age":
            # self.prompt_list = ["A face photo of of a child", "A face photo of of an adult people", "A face photo of of an old people",]
            self.prompt_list = age_prompts
            output_dim=3
        elif self.cfg.classifier_name=="race":
            # self.prompt_list = ["A face photo of of a White people", "A face photo of of a Black people", "A face photo of of an Indian people", "A face photo of of an Asian people",]
            self.prompt_list = race_prompts
            output_dim=4

        self.calculation = {
            t:[0 for _ in range(len(self.prompt_list))] 
            for t in range(*self.cfg.locate_timestep_range)
            }
        
        self.feature_value = {
            t:[0 for _ in range(len(self.prompt_list))] 
            for t in range(*self.cfg.locate_timestep_range)
            }

        classification_model_path = os.path.join(self.cfg.classification_model_path)
        self.h_classification = make_model(classification_model_path, h_classification_widths[self.local_hookpoints()[0]], output_dim=output_dim)
        self.h_classification = self.h_classification

        # Inference Timesteps
        self.forward_timesteps = self.model.scheduler.timesteps
        self.inversion_timesteps = list(reversed([self.forward_timesteps[k] for k in range(1, 50) if k < (50-1)]))

        # Generator for Latents
        self.generator = torch.Generator(device=device)

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

        device = get_model_device(self.model.unet)
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
            total=num_batches,
        )
        
        hidden_dict: dict[str, Tensor] = {}
        name_to_module = {
            name: self.model.unet.get_submodule(name) for name in self.cfg.hookpoints
        }

        maybe_wrapped = {}
        module_to_name = {v: k for k, v in name_to_module.items()}

        def hook(module: nn.Module, _, outputs):
            # Maybe unpack tuple outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            name = module_to_name[module]
            
            outputs = outputs.contiguous()

            # We just use the conditional bottleneck
            # hidden_dict[name].append(outputs[outputs.size(0)//2:, ...])

            hidden_dict[name].append(outputs)

        for num, _ in enumerate(dl):
            hidden_dict.clear()
            hidden_dict = {name: [] for name in self.cfg.hookpoints}

            # Forward pass on the model to get the next batch of activations            
            handles = [
                mod.register_forward_hook(hook) for mod in name_to_module.values()
            ]

            image_latents = torch.randn(
                    (1, 4, 512 // 8, 512 // 8),
                    generator = self.generator,
                    device = torch.device("cuda:0")
                ).to(torch.float16)
            
            image_latents = torch.cat([image_latents, image_latents], dim = 0)
            
            occ = random.choice(occupations)

            prompt = np.random.choice(self.prompt_list, size=2, replace=False).tolist()
            
            # DONT USE OCCUPATIONS
            # prompt = [_.replace("person", occ) for _ in prompt]

            batch = prompt
            class_id = [self.prompt_list.index(_.replace(occ, "person")) for _ in batch[:1]]

            print(f"prompt: {prompt} class_id: {class_id}")

            try:
                with torch.no_grad():
                    image = self.model(batch, latents = image_latents, eta=0.0).images[0]
                    image.save("./vis.png")
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
                
                for timestep_idx in range(*self.cfg.locate_timestep_range):
                    if self.forward_timesteps[timestep_idx] not in self.inversion_timesteps:
                        continue

                    with torch.no_grad():
                        # Don't use the first block
                        reversed_timestep_idx = self.inversion_timesteps.index(self.forward_timesteps[timestep_idx])

                        # unused
                        # target_attribute = self.h_classification(hiddens[timestep_idx], reversed_timestep_idx)/temperature 
                        # We use prompt class to replace h classification
                        # target_attribute = torch.argmax(target_attribute, dim=1)
                        target_attribute = class_id
                        
                    
                    wrapped = maybe_wrapped[name][0]
                    pre_acts = wrapped.pre_acts(hiddens[timestep_idx])
                    top_acts, top_indices = wrapped.select_topk(pre_acts)
                    batch_size, k, height, width = top_acts.shape

                    sparse_latents = torch.zeros((batch_size, wrapped.num_latents, height, width), 
                                        device=wrapped.device, dtype=wrapped.dtype)
                    
                    sparse_latents.scatter_(1, top_indices, top_acts)

                    scaled_weights, weights_step = scaled_input_pair_baseline(sparse_latents.detach(), batch_size = self.cfg.scale_batch_size, num_batch = self.cfg.num_batch)
                    current_time_grad = None        
                    
                    for batch_idx in range(self.cfg.num_batch):
                        batch_weights = scaled_weights[batch_idx * self.cfg.scale_batch_size:(batch_idx + 1) * self.cfg.scale_batch_size].squeeze(0)
                        batch_weights.requires_grad = True
                        batch_weights.retain_grad()
                        temperature = 1000.
                        
                        recons = wrapped.decoder(batch_weights)
                        
                        logits = self.h_classification(recons, reversed_timestep_idx)/temperature
                        logits = torch.nn.functional.softmax(logits, dim=1)
                        
                        logits[:, target_attribute].backward()
                        
                        if timestep_idx > 47:
                            print(logits[:, target_attribute])
                        
                        grad = batch_weights.grad

                        current_time_grad = current_time_grad+grad if current_time_grad is not None else grad
                    
                    integrated_gradients = (current_time_grad*weights_step)

                    top_indices = torch.cat([top_indices[0].unsqueeze(0), top_indices[2].unsqueeze(0)], dim=0)
                    results = torch.gather(integrated_gradients, 1, top_indices[1:]).cpu().to(torch.float16)                    

                    # We save another version, with abs function on weights step
                    integrated_gradients_abs = (current_time_grad*torch.abs(weights_step))
                    self.calculation[timestep_idx][target_attribute[0]] += integrated_gradients_abs
                    self.feature_value[timestep_idx][target_attribute[0]] += weights_step
                    # results_abs = torch.gather(integrated_gradients, 1, top_indices).cpu().to(torch.float16)                    

                    for macro_k, (_result, _target, _topk_indices) in enumerate(zip(results, target_attribute * 2, top_indices)):
                        save_path = os.path.join(self.cfg.save_path,
                                                f"kernel_{self.cfg.kernel_size}",
                                                f"k_{self.cfg.ae_k}", 
                                                f"expansion_factor_{self.cfg.ae_expansion_factor}", 
                                                f"{self.cfg.classifier_name}",
                                                f"timestep_{timestep_idx}",
                                                f"target_{int(_target)}")
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(_result, os.path.join(save_path, f"{num}_{macro_k}.pt"))
                        torch.save(_topk_indices, os.path.join(save_path, f"{num}_{macro_k}_indices.pt"))
                    
                    if num%100 == 99:
                        for attr in range(len(self.calculation[timestep_idx])):
                            print(self.calculation[timestep_idx][attr].mean())
                            save_path = os.path.join(self.cfg.save_path,
                                                    f"kernel_{self.cfg.kernel_size}",
                                                    f"k_{self.cfg.ae_k}", 
                                                    f"expansion_factor_{self.cfg.ae_expansion_factor}", 
                                                    f"{self.cfg.classifier_name}",
                                                    f"timestep_{timestep_idx}",
                                                    f"target_{int(attr)}")
                            os.makedirs(save_path, exist_ok=True)
                            torch.save(self.calculation[timestep_idx][attr].detach().cpu(), os.path.join(save_path, f"calculation.pt"))

                    if num%100 == 99:
                        for attr in range(len(self.feature_value[timestep_idx])):
                            print(self.feature_value[timestep_idx][attr].mean())
                            save_path = os.path.join(self.cfg.save_path,
                                                    f"kernel_{self.cfg.kernel_size}",
                                                    f"k_{self.cfg.ae_k}", 
                                                    f"expansion_factor_{self.cfg.ae_expansion_factor}", 
                                                    f"{self.cfg.classifier_name}",
                                                    f"timestep_{timestep_idx}",
                                                    f"target_{int(attr)}")
                            os.makedirs(save_path, exist_ok=True)
                            torch.save(self.feature_value[timestep_idx][attr].detach().cpu()/float(num), os.path.join(save_path, f"feature_value.pt"))
                            
                        
            pbar.update()

        pbar.close()