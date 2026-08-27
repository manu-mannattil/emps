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

# plot_psi("uniaxial_perpendicular_growth_0.64_2.7", "uniphase_d.png")
# plot_psi("uniaxial_parallel_lamella_0.84_3.18", "uniphase_b.png")
# plot_psi("uniaxial_perpendicular_lamella_1.36_3.1", "uniphase_c.png")
# plot_psi("uniaxial_perpendicular_lamella_0.91_1.77", "uniphase_e.png")
# plot_psi("uniaxial_parallel_lamella_1.1_1.54", "uniphase_f.png")
# plot_psi("uniaxial_parallel_growth_1.5_1.45", "uniphase_g.png")

plot_psi("uniaxial_crit_0.2_0.67_4", "uniphase2_b.png")
plot_psi("uniaxial_offcrit_0.58_0.67_4", "uniphase2_c.png")
plot_psi("uniaxial_crit_0.2_1.5_4", "uniphase3_b.png")
plot_psi("uniaxial_offcrit_0.67_1.5_4", "uniphase3_c.png")
