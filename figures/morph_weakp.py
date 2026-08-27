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

for phi_0 in [0.61, 0.83]:
    for num in range(7):
        plot_psi(f"weakp_{phi_0}_{num}", f"weakp_{phi_0}_{num}.png")
