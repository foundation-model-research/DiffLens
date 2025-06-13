from .P2.P2_generate import P2_main
from .stable_diffusion.SD_generate import SD_main

def bias_mitigation_func(main_args):
    assert main_args.model_name in ["P2", "Stable Diffusion v1.5"], "Choose model from 'P2' or 'Stable Diffusion v1.5'"
    
    if main_args.model_name=="P2":
        return P2_main
    else:
        return SD_main