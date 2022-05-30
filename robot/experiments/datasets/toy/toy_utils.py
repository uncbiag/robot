import numpy as np
import torch
from random import choices
from imageio import imread
import matplotlib.pyplot as plt


def load_image(fname):
    img = imread(fname, as_gray=True)  # Grayscale
    img = (img[::-1, :]) / 255.0
    return 1 - img


def draw_samples(fname, n, dtype=torch.FloatTensor):
    A = load_image(fname)
    xg, yg = np.meshgrid(np.linspace(0, 1, A.shape[0]), np.linspace(0, 1, A.shape[1]))

    grid = list(zip(xg.ravel(), yg.ravel()))
    dens = A.ravel() / A.sum()
    dots = np.array(choices(grid, dens, k=n))
    dots += (0.5 / A.shape[0]) * np.random.standard_normal(dots.shape)

    return torch.from_numpy(dots).type(dtype)


def display_samples(x, color=[(0.55, 0.55, 0.95)]):
    fig = plt.figure()
    x_ = x.detach().cpu().numpy()
    plt.scatter(x_[:, 0], x_[:, 1], 25 * 500 / len(x_), color, edgecolors="none")
    fig.show()
