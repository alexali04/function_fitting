import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import Adam

from typing import List, Callable, Any, Tuple
from torch.optim.optimizer import (
    Optimizer
)

from func_learning.utils.plotting_utils import (
    plot, plot_scatter
)
from tqdm import tqdm

def simple_train_gif(
    model: nn.Module, 
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

        

def double_train_gif(
    model: nn.Module, 
    x_train_pos,
    y_train_pos,
    x_train_neg,
    y_train_neg,
    criterion: Callable = MSELoss, 
    optimizer: Optimizer = Adam,
    epochs: int = 10,
    FOLDER: str = ""
):
    """
    Runs experiment to do structured prediction - firs
    """
    i = 0

    for _, epoch in enumerate(tqdm(range(epochs))):
        model.train()
        optimizer.zero_grad()

        outputs = model(x_train_pos)
        loss = criterion(outputs, y_train_pos)

        loss.backward()
        optimizer.step()

        plot_scatter(outputs, x_train_pos, y_train_pos, i, FOLDER)
        i += 1
    
    for _, epoch in enumerate(tqdm(range(epochs))):
        model.train()
        optimizer.zero_grad()

        outputs = model(x_train_neg)
        loss = criterion(outputs, y_train_neg)

        loss.backward()
        optimizer.step()

        
        plot_scatter(outputs, x_train_neg, y_train_neg, i, FOLDER)
        i += 1

    
    



        



            