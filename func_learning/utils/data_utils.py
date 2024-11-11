import numpy as np
import torch

def generate_circle_data(
    num_points: int = 5000,
    low: float = 0.0,
    high: float = 10.0
    ):
    """
    Function is (sin(t) + 3, cos(t) + 3)
    """

    t = np.linspace(-10, 10, 1000).reshape(-1, 1).astype(np.float32)
    x_train = (np.sin(t) + 3.0).astype(np.float32)
    y_train = (np.cos(t) + 3.0).astype(np.float32)


    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)
    
    return x_train_tensor, y_train_tensor


def generate_cos_data(
    num_points: int = 5000,
    low: float = 0.0,
    high: float = 10.0
    ):
    """
    Function is (sin(t) + 3, cos(t) + 3)
    """

    x_train = np.linspace(-10, 10, 1000).reshape(-1, 1).astype(np.float32)
    y_train = (2.0 * np.cos(x_train) + 3.0).astype(np.float32)

    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)
    
    return x_train_tensor, y_train_tensor