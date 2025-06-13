import os
import torch as th
from tqdm import tqdm

ATTR_NAME_MAP = {
    "age": {
        "0": "young",
        "1": "adult",
        "2": "old",
    },
    "gender": {
        "0": "female",
        "1": "male",
    },
    "race": {
        "0": "white",
        "1": "black",
        "2": "asian",
        "3": "indian",
    }
}

def load_calculation_ignore_timesteps(
    features_saved_path = "../saved",
    attribution_name = "age",
    attribution_id = "1", 
    timestep_range: list[int]=[1, 2], 
    aggregate_saved_path = "./save"):
    
    saved_name = ATTR_NAME_MAP[attribution_name][attribution_id]

    target_attr_list = []
    all_attr_list = []

    calculation_ignore_timesteps = 0.0
    for t in tqdm(range(*timestep_range)):
        base_path = os.path.join(features_saved_path, f"timestep_{t}/target_{attribution_id}/calculation.pt")
        
        if not os.path.exists(base_path):
            continue
        
        calculation = th.load(base_path, map_location="cpu")
        
        calculation = calculation[0]
        calculation = calculation.sum(dim = (-1, -2))
        
        calculation_ignore_timesteps += calculation
    
    value, indices = th.topk(calculation_ignore_timesteps, 100)

    th.save(indices, os.path.join(aggregate_saved_path, f'{saved_name}.pt'))

    print(f"values at {t}:", value)
    print(f"indices at {t}:", indices)

if __name__ == "__main__":
    features_saved_path = "./features/P2"
    load_calculation_ignore_timesteps(features_saved_path)