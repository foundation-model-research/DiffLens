"""Tools for tokenizing and manipulating text datasets."""

import os
import random
import math
from multiprocessing import cpu_count
from typing import TypeVar, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase

import glob
import tqdm

T = TypeVar("T", bound=Union[Dataset, DatasetDict])

PROMPT_LIST = [
    "A photo of a male", "A photo of a female",
    "A photo of a old people", "A photo of a child people", "A photo of a adult people",
    "A photo of a White people", "A photo of a Black people", "A photo of a Asian people", "A photo of a Indian people",
    "A photo of a people wearing glasses",
    "A photo of a cook", 
    "A photo of a crane operator", "A photo of a an announcer",
    "A photo of a drafter", "A photo of a doctor", "A photo of a construction worker",
    "A photo of a biologist", "A photo of a chemist", "A photo of a software developer",
    "A photo of a pharmacist", "A photo of a chef", "A photo of a computer programmer",
    "A photo of a security guard", "A photo of a special ed teacher",
    "A photo of a chief executive officer", "A photo of a librarian",
    "A photo of a bartender", "A photo of a primary school teacher",
    "A photo of a pilot", "A photo of a police officer", "A photo of a housekeeper",
    "A photo of a bus driver", "A photo of a childcare worker",
    "A photo of a receptionist", "A photo of a nurse"
]

def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = cpu_count() // 2,
    text_key: str = "text",
    max_seq_len: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
) -> T:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_seq_len` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_seq_len: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.

    Returns:
        The chunked and tokenized dataset.
    """

    def _tokenize_fn(x: dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        sep = tokenizer.eos_token or "<|endoftext|>"
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output.input_ids[0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size]
                for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        return output

    data = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=512,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    return data.with_format(format, columns=["input_ids"])


def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> list[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names


class MemmapDataset(TorchDataset):
    """Torch Dataset backed by a memory-mapped numpy array."""
    def __init__(
        self,
        data_path: str,
        ctx_len: int,
        max_examples: int = None,
        dtype = np.uint16,
    ):
        mmap = np.memmap(data_path, dtype=dtype, mode="r").reshape(-1, ctx_len)
        self.mmap = mmap[:max_examples]

    def __len__(self):
        return len(self.mmap)

    def __getitem__(self, idx):
        return dict(
            input_ids=torch.from_numpy(self.mmap[idx].astype(np.int64))
        )
    
    def select(self, rng: range) -> "MemmapDataset":
        """Select a subset of the dataset."""
        mmap = MemmapDataset.__new__(MemmapDataset)
        mmap.mmap = self.mmap[rng.start:rng.stop]
        return mmap

    def shard(self, num_shards: int, shard_id: int) -> "MemmapDataset":
        """Split the dataset into `num_shards` and return the `shard_id`-th shard."""
        mmap = MemmapDataset.__new__(MemmapDataset)

        # Split the mmap array into `num_shards` and return the `shard_id`-th shard
        shards = np.array_split(self.mmap, num_shards)
        mmap.mmap = shards[shard_id]
        return mmap

class LatentDataset(TorchDataset):
    """Torch Dataset backed by a memory-mapped numpy array."""
    def __init__(
        self,
        data_path: str,
        max_examples: int = None,
        dtype = np.uint16,
    ):
        self.latent_paths = glob.glob(os.path.join(data_path, "*.pt"))
        random.shuffle(self.latent_paths)
        
        if max_examples is not None:
            self.latent_paths = self.latent_paths[:max_examples]

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx):
        latent_path = self.latent_paths[idx]
        try:
            latent = torch.load(latent_path, map_location=torch.device("cpu")).to(torch.float32)
        except:
            latent = torch.zeros(3, 256, 256).to(torch.float32)
            print(latent_path)
        return latent
    
    def select(self, rng: range) -> "LatentDataset":
        """Select a subset of the dataset."""
        latents = LatentDataset.__new__(LatentDataset)
        latents.latent_paths = self.latent_paths[rng.start:rng.stop]
        return latents

    def shard(self, num_shards: int, shard_id: int) -> "LatentDataset":
        """Split the dataset into `num_shards` and return the `shard_id`-th shard."""
        latents = LatentDataset.__new__(LatentDataset)

        # Split the paths into `num_shards` and return the `shard_id`-th shard
        shard_size = len(self.latent_paths) // num_shards
        start = shard_id * shard_size
        end = start + shard_size if shard_id < num_shards - 1 else len(self.latent_paths)
        latents.latent_paths = self.latent_paths[start:end]
        
        return latents
    
class PromptDataset(TorchDataset):
    """Torch Dataset that returns randomly generated prompts."""

    def __init__(self, prompts_list: list = PROMPT_LIST, max_examples: int = None):
        self.prompts_list = prompts_list
        self.max_examples = max_examples if max_examples is not None else 10000
        self.prompts = [random.choice(prompts_list) for _ in range(self.max_examples)]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

    def select(self, rng: range) -> "PromptDataset":
        """Select a subset of the dataset."""
        prompts = PromptDataset(self.prompts_list, max_examples=len(self.prompts))
        prompts.prompts = self.prompts[rng.start:rng.stop]
        return prompts

    def shard(self, num_shards: int, shard_id: int) -> "PromptDataset":
        """Split the dataset into `num_shards` and return the `shard_id`-th shard."""
        prompts = PromptDataset(self.prompts_list, max_examples=len(self.prompts))
        shard_size = len(self.prompts) // num_shards
        start = shard_id * shard_size
        end = start + shard_size if shard_id < num_shards - 1 else len(self.prompts)
        prompts.prompts = self.prompts[start:end]
        return prompts

    
