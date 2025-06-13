import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from dataclasses import dataclass

from simple_parsing import Serializable


@dataclass
class SaeConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""

    multi_topk: bool = False
    """Use Multi-TopK loss."""

    def update(self, other_dict):
        for key, value in other_dict.items():
            if hasattr(self, key) and isinstance(getattr(self, key), Serializable):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    auxk_loss: Tensor
    """AuxK loss, if applicable."""

    multi_topk_fvu: Tensor
    """Multi-TopK FVU, if applicable."""


class Sae(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SaeConfig,
        device = "cpu",
        dtype: torch.dtype = None,
        *,
        decoder: bool = True,
        h: int = 8, # Bottleneck is hidden*8*8
        w: int = 8, # Bottleneck is hidden*8*8
        kernel_size: int = 1,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in

        self.h = h
        self.w = w

        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.encoder = nn.Conv2d(d_in, self.num_latents, kernel_size=kernel_size,
                                 device=device, dtype=dtype, )

        if kernel_size > 1:
            self.decoder = nn.ConvTranspose2d(
                self.num_latents, d_in, kernel_size=kernel_size, padding=0, device=device, dtype=dtype,
            ) if decoder else None
        
        else: 
            self.decoder = nn.Conv2d(
                self.num_latents, d_in, kernel_size=kernel_size, device=device, dtype=dtype,
            ) if decoder else None
        
        if decoder and self.cfg.normalize_decoder:
            with torch.no_grad():
                if kernel_size > 1:
                    self.decoder.weight.data = self.encoder.weight.data.clone()
                else:
                    self.decoder.weight.data = self.encoder.weight.data.transpose(0, 1).clone()


    @staticmethod
    def load_many(
        name: str,
        local: bool = False,
        layers: list[str] = None,
        device = "cpu",
        *,
        decoder: bool = True,
        pattern: str = None,
    ) -> dict[str, "Sae"]:
        """Load SAEs for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        if local:
            repo_path = Path(name)
        else:
            repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        if layers is not None:
            return {
                layer: Sae.load_from_disk(repo_path / layer, device=device, decoder=decoder)
                for layer in natsorted(layers)
            }
        files = [
            f
            for f in repo_path.iterdir()
            if f.is_dir() and (pattern is None or fnmatch(f.name, pattern))
        ]
        return {
            f.name: Sae.load_from_disk(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.name)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        hookpoint: str = None,
        device = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        # Download from the HuggingFace Hub
        repo_path = Path(
            snapshot_download(
                name,
                allow_patterns=f"{hookpoint}/*" if hookpoint is not None else None,
            )
        )
        if hookpoint is not None:
            repo_path = repo_path / hookpoint

        # No layer specified, and there are multiple layers
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError("No config file found; try specifying a layer.")

        return Sae.load_from_disk(repo_path, device=device, decoder=decoder)

    @staticmethod
    def load_from_disk(
        path,
        device = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SaeConfig(**cfg_dict)

        sae = Sae(d_in, cfg, device=device, decoder=decoder)
        load_model(
            model=sae,
            filename=str(path / "sae.safetensors"),
            device=str(device),
            # TODO: Maybe be more fine-grained about this in the future?
            strict=decoder,
        )
        return sae

    def save_to_disk(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                },
                f,
            )

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype
    
    """
    The following code is selecting topk features on c features
    """
    def pre_acts(self, x: Tensor) -> Tensor:
        sae_in = x.to(self.dtype)
        out = self.encoder(sae_in)
        return nn.functional.relu(out)
    
    def select_topk(self, latents: Tensor) -> EncoderOutput:
        """Select the top-k latents for each position."""
        top_acts, top_indices = latents.topk(self.cfg.k, dim=1, sorted=False)
        return EncoderOutput(top_acts, top_indices)

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents for each position."""
        return self.select_topk(self.pre_acts(x))
    
    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.decoder is not None, "Decoder weight was not initialized."
        batch_size, k, height, width = top_acts.shape
        
        sparse_latents = torch.zeros((batch_size, self.num_latents, height, width), 
                                     device=self.device, dtype=self.dtype)
                
        # Use scatter_ to place top_acts into sparse_latents
        sparse_latents.scatter_(1, top_indices, top_acts)
        
        y = self.decoder(sparse_latents)
        
        return y
    
    def forward(self, x: Tensor, dead_mask: Tensor = None) -> ForwardOutput:
        pre_acts = self.pre_acts(x)

        # Decode and compute residual
        top_acts, top_indices = self.select_topk(pre_acts)
        
        sae_out = self.decode(top_acts, top_indices)

        e = sae_out - x

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (x - x.mean(dim=(0,2,3)).reshape(1, -1, 1, 1)).pow(2).to(torch.float32).sum()

        # Second decoder pass for AuxK loss
        # We pass auxk loss right now
        auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).to(torch.float32).sum()

        if torch.isnan(l2_loss):
            print("l2_loss is nan")
            l2_loss = (x * 0.0).sum()

        if torch.isnan(total_variance):
            print("total_variance is nan")
            l2_loss = (x * 0.0).sum()
            total_variance = x.new_tensor(1.0)

        fvu = l2_loss / total_variance
        
        fvu = fvu.to(e.dtype)

        if self.cfg.multi_topk:
            top_acts, top_indices = pre_acts.topk(4 * self.cfg.k, sorted=False)
            sae_out = self.decode(top_acts, top_indices)

            multi_topk_fvu = (sae_out - x).pow(2).to(torch.float32).sum() / total_variance
            multi_topk_fvu = multi_topk_fvu.to(sae_out.dtype)
        else:
            multi_topk_fvu = sae_out.new_tensor(0.0)

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
            multi_topk_fvu,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.decoder is not None, "Decoder weight was not initialized."

        if isinstance(self.decoder, nn.Conv2d):
            self.decoder.weight.data = self.decoder.weight.data.to(torch.float32)
            eps = torch.finfo(self.decoder.weight.dtype).eps
            norm = torch.norm(self.decoder.weight.data.contiguous().view(-1, self.num_latents), dim=0, keepdim=True)
            norm = norm.view(1, self.num_latents, 1, 1)
            self.decoder.weight.data /= norm + eps
            self.decoder.weight.data = self.decoder.weight.data.to(self.dtype)

        elif isinstance(self.decoder, nn.ConvTranspose2d):
            self.decoder.weight.data = self.decoder.weight.data.to(torch.float32)
            eps = torch.finfo(self.decoder.weight.dtype).eps
            norm = torch.norm(self.decoder.weight.data.contiguous().view(self.num_latents, -1), dim=1, keepdim=True)
            norm = norm.view(self.num_latents, 1, 1, 1)
            self.decoder.weight.data /= norm + eps
            self.decoder.weight.data = self.decoder.weight.data.to(self.dtype)

        else:
            raise NotImplementedError

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.decoder is not None, "Decoder weight was not initialized."
        assert self.decoder.weight.grad is not None, "Decoder weight grad is None."

        # Reshape decoder weights and gradients
        W_dec = self.decoder.weight.view(self.d_in, self.num_latents, -1)
        W_dec_grad = self.decoder.weight.grad.view(self.d_in, self.num_latents, -1)

        # Calculate parallel component
        parallel_component = einops.einsum(
            W_dec_grad, W_dec,
            "d_in n_lat hw, d_in n_lat hw -> n_lat hw"
        )

        # Subtract parallel component from gradient
        W_dec_grad -= einops.einsum(
            parallel_component, W_dec,
            "n_lat hw, d_in n_lat hw -> d_in n_lat hw"
        )

        # Reshape gradient back to original shape
        self.decoder.weight.grad = W_dec_grad.view(self.decoder.weight.shape)
