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

import torchvision
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
        torch.load(model_args.model_path, map_location=device_map)
    )

    model.to(device_map)
    model.eval()

    return model, diffusion

if __name__ == "__main__":
    P2_model_args_path = "../../config/P2/P2_model_config/celeba_hq.yaml"
    target_attr = "gender"
    fairface_model_name_or_path = "res34_fair_align_multi_7_20190809.pt"
    latents_dataset_path = "./h_classifier_latents"

    transform = torch.nn.Sequential(
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    
    # FairFace classifier
    fairface_model_name_or_path = fairface_model_name_or_path
    fairface = torchvision.models.resnet34(pretrained=False)
    fairface.fc = nn.Linear(fairface.fc.in_features, 18)
    fairface.load_state_dict(torch.load(fairface_model_name_or_path))
    fairface = fairface.to("cuda").to(torch.float16)
    fairface.eval()

    model, diffusion = load_artifacts_P2(P2_model_args_path, rank=0)
    model.eval()

    if target_attr == "gender":
        labels = ['male', 'female']
        num_list = [1000, 1000]
    elif target_attr == "race":
        # labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
        labels = ['White', 'Black', 'Asian', 'Indian']  # Updated race categories
        num_list = [1000, 1000, 1000, 1000]
    elif target_attr == "age":
        # labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        labels = ['Young', 'Adult', 'Old']  # Updated age groups
        num_list = [1000, 1000, 1000]
    else:
        raise NotImplementedError

    target_num = dict(zip(labels, [0 for _ in labels]))
    # target_num = dict(zip(labels, num_list))

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

        if all([_ == 1000 for _ in target_num.values()]):
            break
        
        sample = diffusion.ddim_sample_loop(
            model,
            (1, 3, 256, 256)
        )

        
        _saved_list = torch.cat([_.unsqueeze(0) for _ in saved_list], dim=0)

        sample = (sample + 1) * 0.5
        sample = sample.contiguous()
            
        image_ = transform(sample)
        image_ = image_.view(-1, 3, 224, 224).to(torch.float16)  
        outputs = fairface(image_)
            
        if target_attr == "gender":
            gender_outputs = outputs[:, 7:9]
            max_indices = torch.argmax(gender_outputs, dim=1)
            
            target_labels = [labels[i] for i in max_indices]
        
        elif target_attr == "race":
            race_outputs = outputs[:, :7]
            max_indices = torch.argmax(race_outputs, dim=1)

            # Map East Asian and Southeast Asian to Asian
            mapped_labels = ['White', 'Black', 'Latino_Hispanic', 'Asian', 'Asian', 'Indian', 'Middle_Eastern']
            target_labels = [mapped_labels[i] for i in max_indices]
            target_labels = [label for label in target_labels if label in labels]

        elif target_attr == "age":
            age_outputs = outputs[:, 9:]
            max_indices = torch.argmax(age_outputs, dim=1)
            
            # Map age groups to Young, Adult, Old
            mapped_labels = ['Young', 'Young', 'Young', 'Adult', 'Adult', 'Adult', 'Adult', 'Old', 'Old']
            target_labels = [mapped_labels[i] for i in max_indices]


        for image_idx, label in enumerate(target_labels):
            if all([_ == 1000 for _ in target_num.values()]):
                break
            if target_num[label] == 1000:
                continue
            else:
                h_vector = _saved_list[:, image_idx].to(torch.float16)
                os.makedirs(os.path.join(latents_dataset_path, label), exist_ok=True)
                torch.save(h_vector, os.path.join(latents_dataset_path, label, f'{target_num[label]}.pt'))
                target_num[label] += 1