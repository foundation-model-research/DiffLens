import argparse
import os

import random
import numpy as np
import torch as th
import torch.distributed as dist

from .guided_diffusion import logger
from .guided_diffusion.unet import ResBlock, AttentionBlock
from .guided_diffusion.sae import Sae, SaeConfig
from .guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from torchvision import utils

from safetensors.torch import load_model

from simple_parsing import Serializable
from typing import List, Optional
import yaml

class GenerateConfig(Serializable):
    # Diffusion process parameters
    learn_sigma: bool = False
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    timestep_respacing: str = ""
    use_kl: bool = False
    predict_xstart: bool = False
    rescale_timesteps: bool = False
    rescale_learned_sigmas: bool = False
    
    # Sampling parameters
    generate_seed: int = 333
    clip_denoised: bool = True
    num_samples: int = 800
    batch_size: int = 16
    use_ddim: bool = False
    model_path: str = ""
    sample_dir: str = ""
    
    # Editing parameters
    top_k: int = 30
    top_k_list: Optional[List[int]] = None
    target_attr: Optional[str] = None
    edit_ratios: Optional[List[float]] = None
    edit_method: str = "multiply_all"
    
    # Sparse Autoencoder parameters
    ae_k: int = 32
    ae_expansion_factor: int = 32
    ae_lr: float = 0.001
    kernel_size: int = 1
    sae_model_path: str = "Your_P2_sae_model_path"
    use_sae: bool = False

    def update(self, other_dict):
        for key, value in other_dict.items():
            if hasattr(self, key) and isinstance(getattr(self, key), Serializable):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)

class ModelConfig(Serializable):
    model_path: str = None
    """Generation model path."""

    dataset_dir: str = None
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

    use_fp16: bool = False
    """Use FP16."""

    use_scale_shift_norm: bool = True
    """Use scale shift norm."""

    def update(self, other_dict):
        for key, value in other_dict.items():
            if hasattr(self, key) and getattr(self, key) is not None:
                continue
            setattr(self, key, value)

def set_generate_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def sample_and_ones_others(edit_ratios):
    ratios = np.array(edit_ratios)
    selected_index = np.random.choice(len(ratios))
    result = np.ones_like(ratios)
    result[selected_index] = ratios[selected_index]
    print(result)
    return result

def load_features(target_attr, top_k=30, top_k_list=None, features_saved_path="features/P2"):
    if target_attr == "age":
        attr_list = ["young", "adult", "old"]
    elif target_attr == "gender":
        attr_list = ["male", "female"]
    elif target_attr == "race":
        attr_list = ['white', 'black', 'asian', 'indian']
    else:
        raise NotImplementedError

    if top_k_list is not None:
        top_k_list = top_k_list.split("_")
        top_k_list = [int(_) for _ in top_k_list]
    else:
        top_k_list = [top_k for _ in attr_list]

    base_path = os.path.join(features_saved_path, target_attr)

    features = []

    assert len(attr_list)==len(top_k_list), "The length of attr_list and top_k_list should be equal."
    for attr, feature_num in zip(attr_list, top_k_list):
        features.append(th.load(os.path.join(base_path, f'{attr}.pt'))[:feature_num])

    return features

def P2_main(main_args):
    args = GenerateConfig()
    with open(main_args.bias_mitigation.generate_config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    args.update(cfg_dict)

    model_args = ModelConfig()
    with open(args.model_args_path, "r") as f:
        model_cfg_dict = yaml.safe_load(f)
    model_args.update(model_cfg_dict)
    model_args.update(model_and_diffusion_defaults())

    sae_cfg = SaeConfig()
    with open(main_args.bias_mitigation.sae_args_path, "r") as f:
        sae_cfg_dict = yaml.safe_load(f)
    sae_cfg.update(sae_cfg_dict)

    set_generate_seed(args.generate_seed)
    
    edit_features = load_features(
        args.target_attr,
        args.top_k,
        args.top_k_list,
        features_saved_path=args.features_saved_path,
    )
    
    # args.edit_ratios = args.edit_ratios.replace("u", "-")
    edit_ratios = [float(_) for _ in args.edit_ratios.split("_")]
    
    edit_method = args.edit_method
    
    # update sae config
    sae_model = Sae(
            d_in = 512,
            cfg = sae_cfg,
            device = th.device("cuda"),
            kernel_size = args.kernel_size,
        )
    sparse_autoencoder_model_path = args.sparse_autoencoder_model_path
    load_model(sae_model, f"{sparse_autoencoder_model_path}/sae.safetensors", device="cuda")

    logger.configure(dir=args.sample_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(model_args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(model_args.model_path, map_location="cpu")
    )

    model.state_dict().keys()

    model.to("cuda")
    if model_args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    count = 0

    while count * args.batch_size < args.num_samples:
        model_kwargs = {}
        if model_args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device="cuda"
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not model_args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, all_samples = sample_fn(
            model,
            shape=(args.batch_size, 3, model_args.image_size, model_args.image_size),
            noise=None,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            # We should use_ddim here
            sae_model = sae_model,
            use_sae = args.use_sae,
            features = edit_features,
            edit_ratios = sample_and_ones_others(edit_ratios) if "probability" in edit_method else edit_ratios,
            edit_method = edit_method,
            # Use deterministic sampling
            eta = 1.0,
            image_idx = count * args.batch_size,
        )

        all_samples = th.cat([_.unsqueeze(0) for _ in all_samples])
        all_samples = all_samples

        # saving png
        for i in range(args.batch_size):
            out_path = os.path.join(logger.get_dir(),
                                    f"{str(count * args.batch_size + i).zfill(5)}.png")
            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        
        count += 1
        # saving npz
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_images.extend([sample.cpu().numpy() for _sample in sample])
        if model_args.class_cond:
            print("conditional sampling")
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images)} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if model_args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    if True:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    logger.log("sampling complete")
