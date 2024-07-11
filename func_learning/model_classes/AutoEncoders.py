import torch.nn as nn
from typing import List, Callable


class AutoEncoder(nn.Module):
    def __init__(self, layer_sizes: List[int], activation: Callable):
        """
        Defines basic auto-encoder

        Args:
            layer_sizes: List of layer sizes growing from largest to smallest. If the inputs are
            [5, 3, 1], the auto-encoder's layer sizes will be [5, 3, 1, 3, 5]. 
            activation: Callable function applied to weighted linear combination of inputs at the end of each layer

        Returns:
            None
        """

        super().__init__()
        self.activation = activation()  

        encoder_layers = []    
        decoder_layers = []

        for i in range(len(layer_sizes) - 2):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            encoder_layers.append(self.activation)
        
        encoder_layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        for i in range(len(layer_sizes) - 1, 1, -1):
            decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i - 1]))
            decoder_layers.append(self.activation)
        
        decoder_layers.append(nn.Linear(layer_sizes[1], layer_sizes[0]))

        self.encoder = nn.Sequential(*encoder_layers)

        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, x):
        encoded = self.encoder(x)
        encode_activation = self.activation(encoded)
        return self.decoder(encode_activation)

