#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import charu
from utils import minmax

rc = {
    "charu.doc": "aps",
    "charu.tex": True,
    "figure.figsize": [1.9 * 246 * charu.pt, 246 / 1.6 * charu.pt],
    "charu.tex.font": "fourier",
}

def plot_psi(name, figure):
    fname = f"../data/{name}.npy"

    fig, ax = plt.subplots()
    ax.set_box_aspect(1)
    ax.set_axis_off()
    plt.axis('off')

    psi = np.load(fname)
    x = np.linspace(-1, 1, psi.shape[0])
    X, Y = np.meshgrid(x, x)
    ax.pcolormesh(X, Y, psi, cmap="RdBu", rasterized=True)

    plt.savefig(figure, pad_inches=0, facecolor="none", crop=True)
    print(f"saved {figure}")
    ax.clear()

plot_psi("comparison_0.2_0", "png/comparison_0.2_0.png")
plot_psi("comparison_0.2_1", "png/comparison_0.2_1.png")
plot_psi("comparison_0.2_2", "png/comparison_0.2_2.png")
plot_psi("comparison_0.2_3", "png/comparison_0.2_3.png")
plot_psi("comparison_0.5_0", "png/comparison_0.5_0.png")
plot_psi("comparison_0.5_1", "png/comparison_0.5_1.png")
plot_psi("comparison_0.5_2", "png/comparison_0.5_2.png")
plot_psi("comparison_0.5_3", "png/comparison_0.5_3.png")
