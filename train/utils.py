import torch
import math
import numpy as np
from typing import Union, List, Tuple
from collections.abc import Iterable

def get_trainable_params(model) -> Tuple[List,int]:
    all_params = model.parameters()
    trainable_params = [p for p in all_params if p.requires_grad == True]
    num_trainable_params = sum(p.numel() for p in trainable_params)

    return trainable_params, num_trainable_params

def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients to have max L2 norm.
    """
    params_with_grad = [p for p in parameters if p.grad is not None]
    
    if len(params_with_grad) == 0:
        return
    
    # compute total L2 norm across all parameter gradients
    total_norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in params_with_grad))
    
    # clip if needed
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params_with_grad:
            p.grad.mul_(clip_coef)

def get_lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, 
                           warmup_iters: int, cosine_cycle_iters: int) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    """
    # linear warmup
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    
    # after cosine cycle, return min lr
    if it >= warmup_iters + cosine_cycle_iters:
        return min_learning_rate
    
    # cosine annealing
    progress = (it - warmup_iters) / cosine_cycle_iters
    return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress))

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of sequences from the dataset.
    Returns input sequences and their corresponding labels (shifted by 1).
    """
    max_start_idx = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    x = torch.stack([torch.from_numpy(dataset[i:i+context_length].astype(np.int64)) for i in start_indices])
    y = torch.stack([torch.from_numpy(dataset[i+1:i+context_length+1].astype(np.int64)) for i in start_indices])
    
    x = x.to(device)
    y = y.to(device)
    
    return x, y

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Numerically stable softmax.
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def save_checkpoint(model: torch.nn.Module, optimizer, iteration: int, path: str):
    """
    Save model, optimizer state, and iteration to a checkpoint file.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state if hasattr(optimizer, 'state') else optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, path)

def load_checkpoint(model: torch.nn.Module, optimizer, path: str) -> int:
    """
    Load model, optimizer state from a checkpoint file.
    Returns the iteration number.
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if hasattr(optimizer, 'state'):
        optimizer.state = checkpoint['optimizer_state_dict']
    else:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['iteration']