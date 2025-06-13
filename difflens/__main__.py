import torch
import numpy as np
import random

import os

import yaml
from argparse import Namespace

from .dissecting.train_sae.train_sae import train_sae_main
from .train_h_classifier import h_train_func
from .locate.locate_features import locate_features_main
from .bias_mitigation import bias_mitigation_func

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_yaml_config(config_file):
    """Load a YAML config file and convert it to a nested Namespace."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    def dict_to_namespace(d):
        namespace = Namespace()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(namespace, key, dict_to_namespace(value))
            else:
                setattr(namespace, key, value)
        return namespace

    return dict_to_namespace(config)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_yaml>")
        sys.exit(1)

    print("Config path: ", sys.argv[1])
        
    args = load_yaml_config(sys.argv[1])

    if args.train_sae.train == True:
        train_sae_main(args)

    if args.train_h_classifier.train == True:
        train_h_classfier = h_train_func(args)
        train_h_classfier(args)

    if args.locate.locate == True:
        locate_features_main(args)

    if args.bias_mitigation.generate == True:
        bias_mitigation_main = bias_mitigation_func(args)
        bias_mitigation_main(args)