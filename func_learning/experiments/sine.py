"""Experiment - Breadth vs Depth"""

import torch.nn as nn
import numpy as np
import os
import sys

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(module_path)


from func_learning.model_classes.BaseNNModel import (
    BasicMLP
)

from func_learning.utils.preprocessing_utils import (
    gen_function_values, 
    prep_data
)

from func_learning.utils.nn_utils import (
    train_model,
    evaluate_model_on_test_set
)

from func_learning.utils.plotting_utils import (
    make_gif
)

# initialize models
#deep_learner = BasicMLP([1] + [2] * 10 + [1], nn.ReLU)
wide_learner = BasicMLP([1] + [10] * 2 + [1], nn.ReLU)
print(wide_learner)

# constants
FUNC_TO_LEARN = np.sin
PATH_TO_WRITE = "func_learning/experiments/experiment_images/sin_images"

X_train, y_train = gen_function_values(
    function=FUNC_TO_LEARN,
    start=-100,
    end=100,
    num_samples=100000
)

X_val, y_val = gen_function_values(
    function=FUNC_TO_LEARN,
    start=-150,
    end=-100,
    num_samples=10000
)

X_test, y_test = gen_function_values(
    function=FUNC_TO_LEARN,
    start=100,
    end=150,
    num_samples=10000
)

train_loader = prep_data(
    X=X_train,
    y=y_train
)

val_loader = prep_data(
    X=X_val,
    y=y_val
)

train_model(
    model=wide_learner,
    criterion=nn.MSELoss,
    n_epochs=1000,
    train_loader=train_loader,
    val_loader=val_loader,
    function_description="sin(x)",
    func_to_learn=FUNC_TO_LEARN,
    folder=PATH_TO_WRITE,
    domain_start=-150,
    domain_end=150,
    plot_range=[-5, 5],
    plot=True
)

make_gif(
    folder=PATH_TO_WRITE,
    name="wide_sine_big_domain_low_lr",
    fps=10
)

loss = evaluate_model_on_test_set(
    model=wide_learner,
    X_test=X_test,
    y_test=y_test,
    loss_func=nn.MSELoss,
    loss_func_description="Mean Squared Error",
    func_description="sin(x)",
    folder=None,
    plot=False
)

print(loss)







