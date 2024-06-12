import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Callable, Optional
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import imageio
import os



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


def pipeline(
    model: nn.Module,
    func_to_learn: Callable,
    loss_func: Callable,
    func_to_learn_description: str,
    n_epochs: int = 5000,
    loss_func_name: Optional[str] = "",
):
    """
    full pipeline for running this experiment
    1. generates training, validation, and test data
    2. converts train, val sets into DataLoaders
    3. trains model, creating gifs of validation / training processes
    4. evaluates model on test data


    the test data, training data, and validation data are all of different domains so there should be no "leakage" (but obviously, certain functions are periodic)
    """

    # Generating Data
    X_train, y_train = gen_func(func_to_learn, -10, 10, 10000)
    X_val, y_val = gen_func(func_to_learn, -30, -10, 10000)
    X_test, y_test = gen_func(func_to_learn, 10, 30, 10000)

    # Convert arrays to Dataloaders
    train_dataloader = prep_data(X_train, y_train)
    val_dataloader = prep_data(X_val, y_val)

    # Train model
    train(model, loss_func, n_epochs, train_dataloader, val_dataloader, func_to_learn_description)

    # Evaluate model
    eval(model, X_test, y_test, loss_func, loss_func_name, func_to_learn_description)

    # Make gif
    make_gif("Train", "sine_gif", 1/60)



def train(
    model: nn.Module, 
    criterion: Callable,  
    n_epochs: int, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    func_to_learn_description: str,
    func_to_learn: Callable,
    folder: str,
    domain: List[int],
    plot_range: List[int]
) -> None:
    """
    print_epoch_mod: prints training, validation loss every print_epoch_mod epochs \n
    available loss functions: 'mse', 'cross_entropy' \n
    returns validation predictions for each epoch    
    """

    loss_func = criterion()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):

        training_loss = 0.0
        val_loss = 0.0
        
        # Training model
        model.train()
        for X, y in train_loader:
            y_pred = model(X)                       # Generate predictions
            loss = loss_func(y_pred, y)             # Compute Loss
            optimizer.zero_grad()                   # Zero out influence of previously computed gradients
            loss.backward()                         # Compute gradients for all tensors in neural network
            optimizer.step()                        # Update parameters according to gradient (and external factors like LR, momentum)

            training_loss += loss.item()            # Aggregating batch loss  


        X_plot_vals = np.linspace(domain[0], domain[1], 5000)
        y_plot_vals = func_to_learn(X_plot_vals)
        y_pred_vals = model(torch.tensor(X_plot_vals, dtype=torch.float32).unsqueeze(1)).detach().numpy().flatten()

        
        plot(
            X_plot_vals,
            y_plot_vals,
            y_pred_vals,
            folder, 
            f"train_epoch_{epoch}", 
            func_to_learn_description,
            epoch,
            False,
            4000,
            plot_range
            )
        

        # Validating model
        model.eval()                                    # Disables train-only features (i.e. dropout)
        with torch.no_grad():                           # Turn off gradient computations
            for X_val, y_val in val_loader:
                y_val_pred = model(X_val)                      
                loss = loss_func(y_val_pred, y_val)
                val_loss += loss.item()

            if epoch % 100 == 0:
                print(f"{epoch} Batch Training Loss: {training_loss / len(train_loader)}")
                print(f"{epoch} Batch Validation Loss: {val_loss / len(val_loader)}")
            





def prep_data(X: np.array, y: np.array) -> DataLoader:
    """
    Converts data into a torch DataLoader
    """

    # Unsqueeze: maps torch[64] --> torch[64, 1]
    # Distinction between a single sample w/ 64 features and 64 samples w/ 1 feature each
    
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.float32).unsqueeze(1))      
    return DataLoader(ds, batch_size=64, shuffle=True)



def gen_func(function, start, end, num_samples):
    """
    This applies function to a num_samples input values between start and end and returns [input, outputs]
    """
    X = np.linspace(start, end, num_samples)
    y = function(X)
    return X, y



def eval(model, X_test, y_test, loss_func: Callable, loss_func_name: str, label_description: str, folder: str):
    """
    Evaluates model on test dataset using loss_func
    """
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    loss_func = loss_func()

    y_preds = model(X_test)
    loss = loss_func(y_preds, y_test)
    print(f"{loss_func_name}: Test Loss {loss}")

    plot(
        X_test.detach().numpy(), 
        y_test.detach().numpy(), 
        y_preds.detach().numpy(), 
        folder, 
        "results", 
        label_description, 
        -1,
        True,      
        6000       # want large sample
        )



def plot(
    X_vals, 
    y_vals, 
    y_vals_pred, 
    folder: str, 
    id: str, 
    description: str, 
    epoch: int = -1, 
    ret: bool = False, 
    sample_limit: int = 100,
    plot_range: List[int] = [-3, 3]
):
    """
    plots X_vals, y_vals, and y_vals_pred. 

    description should be describing the function that we are trying to learn e.g. sine
    """

    indices = np.random.choice(len(X_vals), sample_limit, replace=False)
    indices = indices[np.argsort(indices)]
    X_vals = X_vals[indices]
    y_vals = y_vals[indices]
    y_vals_pred = y_vals_pred[indices]



    plt.figure()
    plt.plot(X_vals, y_vals, label=description)
    plt.plot(X_vals, y_vals_pred, label="predictions")
    if epoch != -1:
        plt.title(f"{description} - Epoch: {epoch}")
    else:
        plt.title(f"{description} - Test / Total")
    plt.ylim(plot_range)
    plt.legend(loc="upper right")
    plt.savefig(f"{folder}/{id}.png")

    if ret:
        plt.show()
    
    plt.close()

def extract_num(filename):
    base = os.path.basename(filename)
    number = ''.join(filter(str.isdigit, base))
    return int(number) if number.isdigit() else float('inf')

def make_gif(folder, name, fps):
    """
    makes a gif 'name' through the images in 'folder'
    """
    print("Making gif")
    with imageio.get_writer(f'{name}.gif', mode = 'I', fps = fps, loop=0) as writer:
        for filename in sorted(os.listdir(folder), key=extract_num):
            if filename.endswith('png'):
                image = imageio.imread(folder+"/"+filename)
                writer.append_data(image)

        
    


            




            