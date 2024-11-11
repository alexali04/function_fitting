import numpy as np
from typing import Optional, Union
import matplotlib.pyplot as plt
import os
import logging
import imageio

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


def plot(y_pred, x, y, id, FOLDER):
    plt.figure()

    plt.plot(x.numpy(), y.numpy(), label="targets")
    plt.plot(x.numpy(), y_pred.detach().numpy(), label="predictions", linestyle="--")
    ax = plt.gca()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal', adjustable="box")
    plt.savefig(f"{FOLDER}/{str(id)}.png")
    plt.close()