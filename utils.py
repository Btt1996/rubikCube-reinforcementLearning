# utils.py

import torch
import numpy as np

def to_tensor(x, device):
    """Converts a numpy array to a PyTorch tensor and moves it to the specified device."""
    return torch.from_numpy(x).float().to(device)

def to_numpy(x):
    """Converts a PyTorch tensor to a numpy array."""
    return x.detach().cpu().numpy()

def get_epsilon(step, epsilon_start, epsilon_end, epsilon_decay):
    """Returns the value of epsilon for the given step."""
    return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step * epsilon_decay)

def update_target_model(target_model, model):
    """Updates the target model's weights to match the current model's weights."""
    target_model.load_state_dict(model.state_dict())
