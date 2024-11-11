import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Callable

class ANN(nn.Module):
    def __init__(self, activation_one, activation_two):
        """
        Defines simple ANN
        """
        super().__init__()

        # Learns f: R -> R

        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.act_1 = activation_one
        self.act_2 = activation_two


    
    def forward(self, x):
        x = self.act_1(self.fc1(x))
        x = self.act_2(self.fc2(x))
        x = self.fc3(x)
        return x
    

