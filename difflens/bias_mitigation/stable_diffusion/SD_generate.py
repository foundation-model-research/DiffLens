import argparse
import os

import random
import numpy as np

import torch
from torchvision import utils

from collections import Counter

from safetensors.torch import load_model

from tqdm import tqdm
from glob import glob

from .SD_model.sae import SaeConfig, Sae
from .SD_model.pipeline_stable_diffusion import StableDiffusionPipeline
from .SD_model.unet import unet_forward
from .SD_model.ddim import DDIMScheduler

from simple_parsing import Serializable
from typing import Optional, List

import types

import yaml

class GenerateConfig(Serializable):
    num_samples: int = 500
    batch_size: int = 16
    model_path: str = ""
    sample_dir: str = "./samples"
    generate_seed: int = 0
    
    # Editing parameters
    top_k: int = -1
    top_k_list: Optional[List[int]] = None
    target_attr: str = "gender"
    edit_ratios: Optional[List[float]] = None
    edit_method: str = "add"
    feature_saved_path: Optional[str] = None
    percent: float = 0.9
    interpret_each_feature: bool = False
    
    # Sparse Autoencoder parameters
    ae_k: int = 32
    ae_expansion_factor: int = 32
    ae_lr: float = 0.001
    kernel_size: int = 1
    sae_path: str = ""
    use_sae: bool = False
    train_step: int = 98

    def update(self, other_dict):
        for key, value in other_dict.items():
            if hasattr(self, key) and isinstance(getattr(self, key), Serializable):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)
    
PROMPT_LIST = [
    "A face of a firefighter", "A face of a doctor",
    "A face of a receptionist", "A face of a nurse"
]

def sample_and_zero_others(edit_ratios):
    ratios = np.array(edit_ratios)
    
    selected_index = np.random.choice(len(ratios))
    
    result = np.zeros_like(ratios)
    
    result[selected_index] = ratios[selected_index]
    print(result)
    return result

def sample_and_ones_others(edit_ratios):
    ratios = np.array(edit_ratios)
    
    selected_index = np.random.choice(len(ratios))
    
    result = np.ones_like(ratios)
    
    result[selected_index] = ratios[selected_index]

    print(result)
    
    return result

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def load_features(target_attr, top_k=30, 
                     ae_k=32, ae_expansion_factor=8, saved_path=None, percent=0.95, kernel_size=1, top_k_list=None):
    if target_attr == "age":
        attr_list = ["Young", "Adult", "Old"]
        attr_list_index = [0, 1, 2]
    elif target_attr == "gender":
        attr_list = ["male", "female"]
        attr_list_index = [0, 1]
    elif target_attr == "race":
        attr_list = ['White', 'Black', 'Asian', 'Indian']
        attr_list_index = [0, 1, 2, 3]
    else:
        raise NotImplementedError

    all_topk_list = []
    all_list = []
    
    for attr_id, attr in enumerate(tqdm(attr_list_index)):
        if top_k_list is not None:
            top_k = top_k_list[attr_id]

        target_attr_list = []
        all_attr_list = []

        calculation_ignore_timesteps = 0.0
        for t in tqdm(range(1, 49)):
            base_path = os.path.join(saved_path, f"kernel_{kernel_size}/k_{ae_k}/expansion_factor_{ae_expansion_factor}/{target_attr}/timestep_{t}/target_{attr}/calculation.pt")
        
            calculation = torch.load(base_path, map_location="cpu")

            calculation = calculation[0]
            calculation = calculation.sum(dim = (-1, -2))
            
            calculation_ignore_timesteps += calculation

        value, indices = torch.topk(calculation_ignore_timesteps, max(100, top_k))

        all_attr_list.extend([indices for _ in range(1, 49)])

        print(f"values at {t}:", value)
        print(f"indices at {t}:", indices)

        # target_attr_list.extend([indices[:top_k] for _ in range(1, 49)])
        target_attr_list.extend([indices[:top_k] for _ in range(1, 49)])

        all_list.append(all_attr_list)
        all_topk_list.append(target_attr_list)

    return all_topk_list, all_list

def set_generate_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def SD_main(main_args):
    args = GenerateConfig()
    with open(main_args.bias_mitigation.generate_config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    args.update(cfg_dict)

    set_generate_seed(args.generate_seed)

    if args.top_k_list is not None:
        top_k_list = args.top_k_list.split("_")
        top_k_list = [int(_) for _ in top_k_list]
    else:
        top_k_list = None
    
    load_func = load_features
    
    edit_features, _all_features = load_func(
        args.target_attr,
        args.top_k,
        args.ae_k,
        args.ae_expansion_factor,
        args.feature_saved_path,
        percent=args.percent,
        kernel_size=args.kernel_size,
        top_k_list=top_k_list,
    )

    args.edit_ratios = args.edit_ratios.replace("u", "-")
    edit_ratios = [float(_) for _ in args.edit_ratios.split("_")]
    edit_method = args.edit_method
    
    sae_cfg = SaeConfig()
    # update sae config
    sae_cfg.k = args.ae_k
    sae_cfg.expansion_factor = args.ae_expansion_factor
    sae_cfg.lr = args.ae_lr

    sae_models = Sae(
        d_in = 1280,
        cfg = sae_cfg,
        device = torch.device("cuda:0"),
        kernel_size = args.kernel_size,
    ).to(torch.float16)

    # update model path
    path = args.sae_path
    load_model(sae_models, os.path.join(path, "sae.safetensors"), device="cuda:0")
    sae_models.eval()

    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    pipe.unet.forward = types.MethodType(unet_forward, pipe.unet)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(50)
    
    for prompt in PROMPT_LIST:
        num_samples = 0
        while True:
            if num_samples*args.batch_size >= args.num_samples:
                break
            
            zero_func = sample_and_zero_others if "add" in edit_method else sample_and_ones_others
            images = pipe(
                [prompt] * args.batch_size,
                sae_models = sae_models,
                edit_features = edit_features,
                edit_ratios = zero_func(edit_ratios) if "probability" in edit_method else edit_ratios,
                edit_method = edit_method,
                use_sae = args.use_sae,
                # We use all_features_set to locate area in h*w
                all_features = None # if "replace" in edit_method (for experiments) else None,
            ).images
            
            top_k = args.top_k if args.top_k_list is None else args.top_k_list
            os.makedirs(
                os.path.join(args.sample_dir, 
                                 f"{args.edit_method}",
                                 f"kernel_{args.kernel_size}", f"k_{args.ae_k}", f"expansion_factor_{args.ae_expansion_factor}",
                                 f"features_{top_k}",
                                 f"{args.edit_ratios}",
                                 f"{prompt.replace(' ', '-')}",),
                exist_ok=True,
            )
            for k, image in enumerate(images):
                image.save(
                    os.path.join(args.sample_dir, 
                                 f"{args.edit_method}",
                                 f"kernel_{args.kernel_size}", f"k_{args.ae_k}", f"expansion_factor_{args.ae_expansion_factor}",
                                 f"features_{top_k}",
                                 f"{args.edit_ratios}",
                                 f"{prompt.replace(' ', '-')}",
                                 f"{num_samples*args.batch_size+k}.png"
                                 ))
            num_samples += 1