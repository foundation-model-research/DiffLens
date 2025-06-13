import os
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass, replace
from multiprocessing import cpu_count

import numpy as np
import random
import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from simple_parsing import field, parse
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from .data import chunk_and_tokenize, MemmapDataset, LatentDataset, PromptDataset
from .config import Serializable, TrainConfig, SaeConfig
from .locate_main.locate_main_P2 import P2LocateFeature
from .locate_main.locate_main_SD import SDLocateFeature

from difflens.generation_models.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers import DDIMScheduler

import yaml

from difflens.generation_models.P2.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

@dataclass
class RunConfig(TrainConfig):
    run_args_path: str = "./template.yaml"
    """Where to load the model args."""

    model_args_path: str = "./difflens/dissecting/generation_models/P2/config/celeba_hq.yaml"
    """Where to load the model args."""

    # Mode select
    locate: bool = True
    """Whether to locate neuron."""

    generate: bool = False
    """Whether to generate images."""

    eval: bool = False
    """Whether to evaluate images."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    max_examples: int = 500
    """Maximum number of examples to use for training."""

    # Set sparse autoencoder params.
    n_dirs_local: int = 32768
    """Set n dirs."""

    ae_k: int = 32
    """Set k in k-sparse autoencoder."""

    ae_expansion_factor: int = 8
    """Set expansion_factor in k-sparse autoencoder."""

    auxk: int = 256
    """Set auxk in k-sparse autoencoder."""

    ae_lr: float = 0.001
    """Set lr in k-sparse autoencoder. We need lr to load model."""
    
    # Set Integrated Gradients params
    save_path: str = "./save"
    """Where to save the results args."""

    scale_batch_size : int = 1
    """Set Integrated Gradients batch size."""

    kernel_size: int = 1
    """Use scale shift norm."""

    num_batch: int = 10
    """Set Integrated Gradients num batch."""

    resume: bool = False
    """Whether to try resuming from the checkpoint present at `run_name`."""

    seed: int = 42
    """Random seed for shuffling the dataset."""

    batch_size: int = 1
    """The latent has h and w."""

    macro_batch_size: int = 1
    """One GPU batch size."""

    locate_timestep: int = 0
    """Choose timestep to train. Train timestep should be larger than 0."""

    classification_model_path: str = None
    """Hidden classification model saved path."""

    sparse_autoencoder_model_path: str = None
    """Sparse autoencoder model saved path."""

    locate_timestep_range: list[str] = field(
        default_factory=lambda: [40, 50],
    )

    num_timesteps: int = 50
    """Choose timestep to train."""

    hookpoints: list[str] = field(
        default_factory=lambda: ["middle_block.2"],
    )
    """Hookpoints."""

    data_preprocessing_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""

    dataset_dir: str = None
    """P2 Model dataset dir"""

    sd_model_path: str = None
    """Stable Diffusion model path"""

    def update(self, other_dict):
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

    use_fp16: bool = False
    """Use FP16."""

    use_scale_shift_norm: bool = True
    """Use scale shift norm."""

    def update(self, other_dict):
        for key, value in other_dict.items():
            if hasattr(self, key) and getattr(self, key) is not None:
                continue
            setattr(self, key, value)
    

def load_artifacts_P2(args: RunConfig, rank: int) -> tuple[PreTrainedModel, Dataset]:
    model_args = ModelConfig()
    with open(args.model_args_path, "r") as f:
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

    if model_args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    dataset = LatentDataset(os.path.join(args.dataset_dir, f"{args.locate_timestep}"), args.max_examples)

    return model, diffusion, dataset

def load_artifacts_SD(args: RunConfig, rank: int) -> tuple[PreTrainedModel, Dataset]:
    device_map=f"cuda:{rank}"

    model = StableDiffusionPipeline.from_pretrained(args.sd_model_path, torch_dtype=torch.float16)
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)

    model.to(device_map)
    
    dataset = PromptDataset(max_examples=args.max_examples)

    return model, None, dataset

def locate_features_main(main_args):
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = RunConfig()
    with open(main_args.locate.locate_config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    args.update(cfg_dict)

    sae_args = SaeConfig()
    with open(main_args.locate.sae_args_path, "r") as f:
        sae_cfg_dict = yaml.safe_load(f)
    sae_args.update(sae_cfg_dict)

    args.sae = sae_args

    assert main_args.model_name in ["P2", "Stable Diffusion v1.5"], "Choose model from 'P2' or 'Stable Diffusion v1.5'"
    load_artifacts = load_artifacts_P2 if main_args.model_name == "P2" else load_artifacts_SD
    
    if args.locate:
        for locate_timestep in range(*args.locate_timestep_range):
            args.locate_timestep = locate_timestep
            # Awkward hack to prevent other ranks from duplicating data preprocessing
            if not ddp or rank == 0:
                model, diffusion, dataset = load_artifacts(args, rank)
            if ddp:
                dist.barrier()
                if rank != 0:
                    model, diffusion, dataset = load_artifacts(args, rank)
                dataset = dataset.shard(dist.get_world_size(), rank)

            # Prevent ranks other than 0 from printing
            with nullcontext() if rank == 0 else redirect_stdout(None):
                print(f"Storing model weights in {model.dtype}")

                locater = P2LocateFeature(args, dataset, model, diffusion) if main_args.model_name == "P2" else SDLocateFeature(args, dataset, model)
                locater.locate()
