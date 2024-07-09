import torch.nn as nn
from typing import List, Callable



class MLP(nn.Module):
    def __init__(self, layer_sizes: List[int], activation: Callable):
        """
        defines computational graph

        layer_sizes = [input_size, hidden_layer_1_size, ..., hidden_layer_n_size, output]

        activation function should be a non-linear Callable function
        """

        # Error Catching
        if len(layer_sizes) < 2:
            raise Exception("Neural Network must have at least two layers")
        if min(layer_sizes) < 1:
            raise Exception("Layers must be of size 1 or more")
        

        super().__init__()
        self.activation = activation()



        # Defining Network Topology

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(self.activation)
        
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.network = nn.Sequential(*layers)


    def forward(self, x):
        return self.network(x)