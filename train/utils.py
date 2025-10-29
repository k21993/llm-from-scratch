import torch
from typing import Union, List, Tuple

def get_trainable_params(model) -> Tuple[List,int]:
    all_params = model.parameters()
    trainable_params = [p for p in all_params if p.requires_grad == True]
    num_trainable_params = sum(p.numel() for p in trainable_params)

    return trainable_params, num_trainable_params