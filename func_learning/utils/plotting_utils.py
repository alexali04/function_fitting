import numpy as np
from typing import Optional, Union
import matplotlib.pyplot as plt
import os
import logging
import imageio


def plot_labels_and_predictions(
    inputs: np.ndarray, 
    labels: np.ndarray, 
    predictions: np.ndarray, 
    folder: str, 
    id: str, 
    function_description: str, 
    epoch: int = -1, 
    show_image: bool = False, 
    sample_limit: int = 100,
    save_fig: bool = False,
    static_range: bool = False,
    y_low: Optional[int] = None,
    y_high: Optional[int] = None
) -> None:
    """
    Generates plot of labels and predictions

    Args:
        inputs: x values
        labels: function value of inputs
        predictions: predicted function value of inputs
        folder: where to save image to
        id: name of resultant png file
        function_description: Name of function being approximated i.e. sin(x) or x^2
        epoch: which epoch in training loop. set to -1 if not training
        only_show_image: if False, write image to folder and continue (during training loop). if True, 
        display image (blocking)
        sample_limit: number of inputs and corresponding labels / predictions to sample
        save_fig: if True, save image to folder. 
        static_range: if True, use y_low and y_high as limits on the y-axis
        y_low: bottom limit of plot displayed
        y_high: top limit of plot displayed

    Returns:
        None
    """

    # Randomly sample for plotting
    indices = np.random.choice(len(inputs), sample_limit, replace=False)
    indices = indices[np.argsort(indices)]
    inputs = inputs[indices]
    labels = labels[indices]
    predictions = predictions[indices]


    plt.figure()
    plt.plot(inputs, labels, label=function_description)
    plt.plot(inputs, predictions, label="predictions")


    if epoch != -1:
        plt.title(f"{function_description} - Epoch: {epoch}")
    else:
        plt.title(f"{function_description} - Test / Total")

    # if we want the range of the plot to stay static
    if static_range:
        plt.ylim(y_low, y_high)

    plt.legend(loc="upper right")

    if save_fig:
        plt.savefig(f"{folder}/{id}.png")
    

    if show_image:
        plt.show()
    
    plt.close()


def extract_num(
    filename: str
) -> Union[int, float]:
    """
    Extracts number from filename
    """
    base = os.path.basename(filename)
    number = ''.join(filter(str.isdigit, base))
    return int(number) if number.isdigit() else float('inf')


def make_gif(
    folder: str,
    name: str,
    fps: int
) -> None:
    """
    Construct gif 'name' from the images in 'folder'

    Args:
        folder: path to gif folder
        name: name of gif to make
        fps: frames per second
    """
    logging.info(f"Making {name}.gif from images in {folder}")

    with imageio.get_writer(f'{name}.gif', mode = 'I', fps = fps, loop=0) as writer:
        for filename in sorted(os.listdir(folder), key=extract_num):
            if filename.endswith('png'):
                image = imageio.imread(folder+"/"+filename)
                writer.append_data(image)

        
    
