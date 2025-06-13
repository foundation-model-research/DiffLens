import os
import torch
from torch.utils.data import Dataset, Subset

from glob import glob

import numpy as np

class AttributeLatentDataset(Dataset):
    def __init__(self, data_path, args):
        self.target_attr = args.target_attr
        self.data_path = data_path
        self.attr_map = {
            "gender": ["female", "male"],
            "race": ['White', 'Black', 'Asian', 'Indian'],
            "age": ['young', 'adult', 'old'],
        }
        
        self.categories = self.attr_map[self.target_attr]
        self.all_files = []
        self.labels = []
        
        if (self.target_attr in ['White', 'Black', 'Asian', 'Indian']
            or
            self.target_attr in ['young', 'adult', 'old']):
            
            for idx, category in enumerate(self.categories):
                category_files = glob(os.path.join(data_path, f'{category}/*.pt'))
                self.all_files.extend(category_files)
                label = 0 if category == self.target_attr else 1
                self.labels.extend([label] * len(category_files))
            
            self.categories = [self.target_attr, "others"]

        else:
            for idx, category in enumerate(self.categories):
                category_files = glob(os.path.join(data_path, f'{category}/*.pt'))
                self.all_files.extend(category_files)
                self.labels.extend([idx] * len(category_files))
        
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        latent = torch.load(self.all_files[idx])
        label = self.labels[idx]
        return latent, label
    

def fast_train_test_split(dataset, test_size=0.1, random_state=None):    
    num_samples = len(dataset)
    num_test = int(num_samples * test_size)
    indices = np.arange(num_samples)
    
    if random_state is not None:
        np.random.seed(random_state)
        np.random.shuffle(indices)
    
    train_indices = indices[num_test:]
    test_indices = indices[:num_test]
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
        
    return train_dataset, test_dataset
