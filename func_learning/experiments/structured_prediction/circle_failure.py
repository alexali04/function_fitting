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
    generate_circle_data
)

from func_learning.model_classes.base_nn import (
    ANN
)

from func_learning.utils.plotting_utils import (
    make_gif
)

FOLDER = "func_learning/images/struc_pred"
os.makedirs(FOLDER, exist_ok=True)

circle_model = ANN(activation=torch.sin)
optimizer = SGD(circle_model.parameters(), lr=0.01)
loss = nn.MSELoss()

x_train, y_train = generate_circle_data()

simple_train_gif(circle_model, x_train, y_train, loss, optimizer, 100, FOLDER)
make_gif(folder=FOLDER, name="circle_learning", fps=3)

