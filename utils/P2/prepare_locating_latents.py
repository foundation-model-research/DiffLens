import os
import torch
from torch import nn

from simple_parsing import Serializable

import sys
sys.path.append("../..")
from difflens.generation_models.P2.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

import yaml

class ModelConfig(Serializable):
    model_path: str = None
    """Generation model path."""

    dataset_dir: str = ModuleNotFoundError
    """Latents dataset dir."""

    num_samples: int = 10000
    """Num of images."""

    sample_dir: str = "samples"
    """Save dir of images."""

    clip_denoised: bool = True
    """Whether to clip."""

    batch_size: int = 16
    """Batch size of sampling."""

    use_ddim: bool = True
    """Whether to use DDIM."""

    timestep_respacing: str = '50'
    """DDIM sampling steps."""

    attention_resolutions: str = '16'
    """Attention resolutions."""

    class_cond: bool = False
    """Class conditional."""

    diffusion_steps: int = 1000
    """Diffusion steps."""

    dropout: float = 0.0
    """Dropout rate."""

    image_size: int = 256
    """Image size."""

    learn_sigma: bool = True
    """Learn sigma."""

    noise_schedule: str = "linear"
    """Noise schedule."""

    num_channels: int = 128
    """Number of channels."""

    num_res_blocks: int = 1
    """Number of residual blocks."""

    num_head_channels: int = 64
    """Number of head channels."""

    resblock_updown: bool = True
    """Resblock updown."""

    use_scale_shift_norm: bool = True
    """Use scale shift norm."""

    use_fp16: bool = False

    def update(self, other_dict):
        if other_dict is None:
            return
        for key, value in other_dict.items():
            if hasattr(self, key) and getattr(self, key) is not None:
                continue
            setattr(self, key, value)

def load_artifacts_P2(model_args_path, rank: int):
    model_args = ModelConfig()
    with open(model_args_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    model_args.update(cfg_dict)

    model_args.update(model_and_diffusion_defaults())

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(model_args, model_and_diffusion_defaults().keys())
    )
    
    device_map=f"cuda:{rank}"
    model.load_state_dict(
        torch.load(f"{model_args.model_path}", map_location=device_map)
    )

    model.to(device_map)
    model.eval()

    return model, diffusion

if __name__ == "__main__":
    P2_model_args_path = "../../config/P2/P2_model_config/celeba_hq.yaml"
    target_attr = "gender"
    fairface_model_name_or_path = "res34_fair_align_multi_7_20190809.pt"
    latents_dataset_path = "./forward_latents"

    model, diffusion = load_artifacts_P2(P2_model_args_path, rank=0)
    model.eval()

    target_num = 0

    saved_list = []
    
    def hook_fn(module, input, output):
        saved_list.append(output.detach().cpu())

    hooks = []
    for name, module in model.named_modules():
        if name=="out":
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    while True:
        saved_list.clear()

        sample = diffusion.ddim_sample_loop(
            model,
            (1, 3, 256, 256)
        )

        _saved_list = torch.cat([_ for _ in saved_list], dim=0)

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        
        if target_num == 1000:
            continue
        else:
            h_vector = _saved_list[:, target_num].to(torch.float16)
            for timestep in range(h_vector.size(0)):
                os.makedirs(os.path.join(latents_dataset_path, f"{timestep}"), exist_ok=True)
                torch.save(h_vector, os.path.join(latents_dataset_path, f"{timestep}", f'{target_num}.pt'))
                target_num += 1