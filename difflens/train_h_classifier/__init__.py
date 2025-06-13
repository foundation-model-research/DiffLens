from .trainer_P2 import train_main_P2
from .trainer_SD import train_main_SD

def h_train_func(main_args):
    assert main_args.model_name in ["P2", "Stable Diffusion v1.5"], "Choose model from 'P2' or 'Stable Diffusion v1.5'"
    
    if main_args.model_name=="P2":
        return train_main_P2
    else:
        return train_main_SD