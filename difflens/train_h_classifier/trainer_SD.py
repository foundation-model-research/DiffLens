# Thanks @yongxiang for implementation of SD h-classifier.  
# Modified from @yongxiang's original implementation

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDIMScheduler
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import yaml

from simple_parsing import Serializable

class ClassifierTrainConfig(Serializable):
    model_path: str = None
    """Your stable diffusion model path"""

    train_samples: int = 32
    test_samples: int = 32
    macro_batch: int = 4

    num_epochs: int = 30
    save_dir: str = "./h_classifier"
    attribute: str = "gender"

    def update(self, other_dict):
        if other_dict is None:
            return
        for key, value in other_dict.items():
            if hasattr(self, key) and isinstance(getattr(self, key), Serializable):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)

def setup_pipeline(model_path, device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device)


class DynamicAttributeDataset(Dataset):
    def __init__(self, attribute, prompt_pool, categories, num_samples, pipe, num_timesteps=50, macro_batch_size=4):
        self.attribute = attribute  # 'age', 'gender', or 'race'
        self.prompt_pool = prompt_pool
        self.categories = categories
        self.num_samples = num_samples
        self.num_timesteps = num_timesteps
        self.macro_batch_size = macro_batch_size
        self.pipe = pipe

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get the prompt for the current attribute
        attribute_value, prompt = self.prompt_pool[idx % len(self.prompt_pool)]
        
        # Generate random values for other attributes
        gender = random.choice(self.categories['gender'])
        race = random.choice(self.categories['race'])
        age = random.choice(self.categories['age'])
        
        # Override the target attribute with the selected value
        if self.attribute == 'age':
            age = attribute_value
            label = self.categories['age'].index(attribute_value)
        elif self.attribute == 'gender':
            gender = attribute_value
            label = self.categories['gender'].index(attribute_value)
        elif self.attribute == 'race':
            race = attribute_value
            label = self.categories['race'].index(attribute_value)
        
        combined_prompt = f"A photo of a {gender} {race} {age} person"
        
        # Hook to capture middle block outputs
        middle_block_outputs = []
        
        def hook_fn(module, input, output):
            # Use only the classifier-free guidance part
            middle_block_outputs.append(output[output.size(0)//2:, ...].to(torch.float32).detach().cpu())
        
        hook = self.pipe.unet.mid_block.register_forward_hook(hook_fn)
        middle_block_outputs.clear()
        
        with torch.no_grad():
            _ = self.pipe([combined_prompt]*self.macro_batch_size, 
                         num_inference_steps=self.num_timesteps).images[0]
        
        hook.remove()
        latents = torch.stack(middle_block_outputs)
        return latents, torch.tensor(label, dtype=torch.long)
    

class AttributeClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(AttributeClassifier, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(1280 * 8 * 8, num_classes) for _ in range(48)])
        self.forward_timesteps = list(range(1, 49))
        self.reversed_timesteps = list(reversed(self.forward_timesteps))

    def forward(self, x, t):
        reversed_t = self.reversed_timesteps[self.forward_timesteps.index(t)]
        x_reverse = x[:, reversed_t, ...]
        x_reverse = x_reverse.reshape(-1, *x_reverse.size()[-3:])
        x_reverse = x_reverse.reshape(x_reverse.size(0), -1)
        return self.linears[t - 1](x_reverse)


def train_main_SD(args):
    train_args = ClassifierTrainConfig()

    with open(args.train_h_classifier.h_classifier_config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    train_args.update(cfg_dict)

    categories = {
        "gender": ["male", "female"],
        "age": ["child", "adult", "old"],
        "race": ["White", "Black", "Asian", "Indian"]
    }

    prompt_pool = {
        "gender": [
            ("male", "A photo of a male person"),
            ("female", "A photo of a female person"),
        ],
        "age": [
            ("child", "A photo of a child"),
            ("adult", "A photo of an adult"),
            ("old", "A photo of an old person")
        ],
        "race": [
            ("White", "A photo of a White person"),
            ("Black", "A photo of a Black person"),
            ("Asian", "A photo of an Asian person"),
            ("Indian", "A photo of an Indian person")
        ],
        }

    prompt_pool = prompt_pool[train_args.attribute]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = setup_pipeline(train_args.model_path, device)
    
    train_dataset = DynamicAttributeDataset(
        train_args.attribute, prompt_pool, categories, train_args.train_samples, pipe, macro_batch_size=train_args.macro_batch
    )
    test_dataset = DynamicAttributeDataset(
        train_args.attribute, prompt_pool, categories, train_args.test_samples, pipe, macro_batch_size=train_args.macro_batch
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    model = AttributeClassifier(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7)
    
    train_accuracy_history = []
    test_accuracy_history = []
    
    for epoch in range(train_args.num_epochs):
        model.train()
        train_loss = [0 for _ in range(1, 49)]
        train_acc = [0 for _ in range(1, 49)]
        
        for latents, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_args.num_epochs} - Training"):
            latents, labels = latents.to(device), labels.to(device)
            optimizer.zero_grad()
            
            for time_id, t in enumerate(range(1, 49)):
                outputs = model(latents, t)
                target = labels.unsqueeze(1).repeat(train_loader.dataset.macro_batch_size, 1).view(-1)
                loss = criterion(outputs.view(-1, 3), target)
                loss.backward()
                optimizer.step()
                
                train_loss[time_id] += loss.item()
                train_acc[time_id] += (outputs.argmax(1) == target).sum().item()
        
        train_loss = [l/len(train_loader)/train_loader.dataset.macro_batch_size for l in train_loss]
        train_acc = [a/len(train_loader)/train_loader.dataset.macro_batch_size for a in train_acc]
        train_accuracy_history.append(train_acc)
        
        model.eval()
        test_loss = [0 for _ in range(1, 49)]
        test_acc = [0 for _ in range(1, 49)]
        
        with torch.no_grad():
            for latents, labels in tqdm.tqdm(test_loader, desc=f"Epoch {epoch+1}/{train_args.num_epochs} - Testing"):
                latents, labels = latents.to(device), labels.to(device)
                
                for time_id, t in enumerate(range(1, 49)):
                    outputs = model(latents, t)
                    target = labels.unsqueeze(1).repeat(test_loader.dataset.macro_batch_size, 1).view(-1)
                    loss = criterion(outputs.view(-1, 3), target)
                    
                    test_loss[time_id] += loss.item()
                    test_acc[time_id] += (outputs.argmax(1) == target).sum().item()
        
        test_loss = [l/len(test_loader)/test_loader.dataset.macro_batch_size for l in test_loss]
        test_acc = [a/len(test_loader)/test_loader.dataset.macro_batch_size for a in test_acc]
        test_accuracy_history.append(test_acc)
        
        print(f'Epoch {epoch+1}: train loss {np.mean(train_loss):.4f}, '
              f'train acc {np.mean(train_acc):.4f}, '
              f'test loss {np.mean(test_loss):.4f}, '
              f'test acc {np.mean(test_acc):.4f}')
        
        if epoch % 10 == 0 or epoch == train_args.num_epochs - 1:
            os.makedirs(f"{train_args.save_dir}/{train_args.attribute}", exist_ok=True)
            torch.save(model.state_dict(), f'{train_args.save_dir}/{train_args.attribute}/classifier_epoch_{epoch}.pth')

    torch.save(model.state_dict(), f'{train_args.save_dir}/{train_args.attribute}/model_last_epoch.pth')            
                
    return