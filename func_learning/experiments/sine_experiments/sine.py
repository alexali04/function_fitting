"""Experiment - Breadth vs Depth


Learned: 
When training on toy datasets, it's better to not use data

"""

import os
import sys
import torch.nn as nn
from torch.optim import SGD
import torch

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(module_path)

from func_learning.utils.train_utils import (
    simple_train_gif
)

from func_learning.utils.data_utils import (
    generate_cos_data
)

from func_learning.model_classes.base_nn import (
    ANN
)

from func_learning.utils.plotting_utils import (
    make_gif
)

FOLDER = "func_learning/images/sine_images"
os.makedirs(FOLDER, exist_ok=True)

sin_model = ANN(activation=torch.sin)
optimizer = SGD(sin_model.parameters(), lr=0.01)
loss = nn.MSELoss()

x_train, y_train = generate_cos_data()

simple_train_gif(sin_model, x_train, y_train, loss, optimizer, 150, FOLDER)
make_gif(folder=FOLDER, name="cos_learning", fps=3)










