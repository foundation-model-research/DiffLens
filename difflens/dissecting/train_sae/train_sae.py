# The main code of k-sae is copied and modified from 
# https://github.com/EleutherAI/sparsify

# We train an SAE for all timesteps

import os
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from multiprocessing import cpu_count

import numpy as np
import random
import torch
import torch.distributed as dist
from torch import nn
from datasets import Dataset
from simple_parsing import field, parse

from .trainer_P2 import P2SaeTrainer, TrainConfig
from .config import SaeConfig
from .trainer_SD import SDSaeTrainer
from .config import Serializable
from .data import PromptDataset

import yaml

from difflens.generation_models.P2.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from diffusers import StableDiffusionPipeline, DDIMScheduler

@dataclass
class RunConfig(TrainConfig):
    model_args_path: str = ""
    """Where to load the model args."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    max_examples: int = None
    """Maximum number of examples to use for training."""
    
    # Sometimes we may change the kernel size
    kernel_size: int = 1
    """Set kernel size."""

    lr_warmup_ratio: float = 0.0
    """Set warm-up ratio."""

    lr: float = 0.0001
    """Set learning rate."""

    eval_split: float = 0.2
    """Dataset split ratio to use for eval."""

    evaluate_every: int = 10
    """Evaluation steps."""

    resume: bool = False
    """Whether to try resuming from the checkpoint present at `run_name`."""

    seed: int = 42
    """Random seed for shuffling the dataset."""

    batch_size: int = 4
    """The latent has h and w."""

    macro_batch_size: int = 4
    """One GPU batch size."""

    train_batch_size: int = 65536
    """Real train batch size."""

    image_size: int = 256
    """The latent has h and w."""

    saved_train_timestep: str = None
    """Saved path."""

    train_timestep_range: list[str] = field(
        default_factory=lambda: [1, 50],
    )

    num_timesteps: int = 50
    """Choose timestep to train."""

    hookpoints: list[str] = field(
        default_factory=lambda: ["input_blocks.1.0"],
    )
    """Hookpoints."""

    data_preprocessing_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""

    def update(self, other_dict):
        if other_dict is None:
            return
        for key, value in other_dict.items():
            if hasattr(self, key) and isinstance(getattr(self, key), Serializable):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)

@dataclass
class ModelConfig(Serializable):
    model_path: str = field(default=None)
    """Generation model path."""

    dataset_dir: str = field(default=None)
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

    num_samples: int = 50000
    """Stable Diffusion training samples."""

    def update(self, other_dict):
        if other_dict is None:
            return
        for key, value in other_dict.items():
            if hasattr(self, key) and getattr(self, key) is not None:
                continue
            setattr(self, key, value)
    

def load_artifacts_P2(args: RunConfig, rank: int) -> tuple[nn.Module, Dataset]:
    model_args = ModelConfig().load_yaml(args.model_args_path, drop_extra_fields = False)
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

def load_artifacts_Stable_Diffusion(args: RunConfig, rank: int) -> tuple[nn.Module, Dataset]:
    model_args = ModelConfig().load_yaml(args.model_args_path, drop_extra_fields = False)
    
    model_id = model_args.model_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    
    device_map=f"cuda:{rank}"
    pipe.to(device_map)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # We only use prompt dataset here
    dataset = PromptDataset(max_examples = args.num_samples)

    return pipe, dataset


def train_sae_main(main_args):
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")
    
    args = RunConfig()
    args.sae = SaeConfig()

    with open(main_args.train_sae.run_args_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    args.update(cfg_dict)

    with open(main_args.train_sae.sae_args_path, "r") as f:
        sae_cfg_dict = yaml.safe_load(f)
    args.sae.update(sae_cfg_dict)

    # One GPU batch size
    world_size = torch.distributed.get_world_size() if dist.is_initialized() else 1
    args.macro_batch_size = args.batch_size // world_size

    # We give examples of Stable Diffusion v1.5 and P2.
    assert main_args.model_name in ["P2", "Stable Diffusion v1.5"], "You can only choose model from 'P2' and 'Stable Diffsion v1.5"
    load_artifacts = load_artifacts_P2 if main_args.model_name=="P2" else load_artifacts_Stable_Diffusion
            
    # Awkward hack to prevent other ranks from duplicating data preprocessing
    if not ddp or rank == 0:
        model, diffusion_or_dataset = load_artifacts(args, rank)
    if ddp:
        dist.barrier()
        if rank != 0:
            model, diffusion_or_dataset = load_artifacts(args, rank)

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        print(f"Storing model weights in {model.dtype}")

        trainer = P2SaeTrainer(args, model, diffusion_or_dataset) if main_args.model_name=="P2" else SDSaeTrainer(args, diffusion_or_dataset, model)

        if args.resume:
            trainer.load_state(args.run_name or f"sae-ckpts/{main_args.model_name}/kernel_{args.kernel_size}/k_{args.sae.k}/expansion_factor_{args.sae.expansion_factor}/lr_{args.lr}")

        trainer.fit()