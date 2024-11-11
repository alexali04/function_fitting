import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import Adam

from typing import List, Callable, Any, Tuple
from torch.optim.optimizer import (
    Optimizer
)

from func_learning.utils.plotting_utils import (
    plot
)
from tqdm import tqdm

def simple_train_gif(model: nn.Module, 
          x_train,
          y_train,
          criterion: Callable = MSELoss, 
          optimizer: Optimizer = Adam,
          epochs: int = 10,
          FOLDER: str = "struc_pred_images"
          ):
    """
    Optimizer is base class for all optimizers (Adam, SGD, etc)
    """
    train_loss = 0.0
    i = 0

    for _, epoch in enumerate(tqdm(range(epochs))):
        model.train()
        optimizer.zero_grad()

        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        plot(outputs, x_train, y_train, i, FOLDER)
        i += 1

        
            


        



            