import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from typing import Callable, Tuple

def prep_data(
    X: np.array, 
    y: np.array
) -> DataLoader:
    """
    Converts data into a torch DataLoader
    """

    # Unsqueeze: maps torch[64] --> torch[64, 1]
    # Distinction between a single sample w/ 64 features and 64 samples w/ 1 feature each
    
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.float32).unsqueeze(1))      
    return DataLoader(ds, batch_size=64, shuffle=True)


def gen_function_values(
    function: Callable,
    start: float,
    end: float,
    num_samples: int
) -> Tuple[np.ndarray]:
    """
    This applies function to a num_samples input values between start and end and returns [input, outputs]
    """
    X = np.linspace(start, end, num_samples)
    y = function(X)
    return X, y
