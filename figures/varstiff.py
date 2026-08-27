# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

kappa = 0.013 # interface parameter (kPa um^2)
B = 0.024 # parameter B (kPa um^2; used in calculating end-to-end distance)
L = 25 # box size (um)
phi_c = 0.2 # critical network volume fraction
phi_0 = 0.2 # mean polymer volume fraction

n = 1024
x = np.linspace(-L, L, n)
X, Y = np.meshgrid(x, x)

# Stiffness field.
Y_min, Y_max = 200, 800 # dry elastomer stiffness range.
sigma = 13 # sharpness
C = (Y_max - Y_min)/(1 - np.exp(-L**2/sigma**2))
Y_field = C * np.exp(-X**2/sigma**2) + (Y_max - C)

import charu

rc = {
    "charu.doc": "aps",
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "figure.figsize": [170 * charu.pt, 170 * charu.pt]
}

with plt.rc_context(rc):
    # Expected domain size.
    h = 35 * np.sqrt(3 * B / Y_field[0]) * phi_c**(-1 / 3)
    M = phi_c**(-5 / 3) * Y_field[0] / 3
    gamma = M * h ** 2 / kappa
    size = 2*np.pi*h/np.sqrt(np.log(gamma))

    fig = plt.figure()

    gs = GridSpec(2, 1, hspace=0, height_ratios=[1, 7])

    ax1 = fig.add_subplot(gs[0])

    ax1.pcolormesh(X, Y, Y_field, cmap="Greys", alpha=0.5, rasterized=True)

    ax1.plot(x, size, "k-")
    ax1.set_ylim(size.min() - 0.2, size.max() + 0.2)
    ax1.set_box_aspect(1/7)

    ax1.minorticks_off()
    ax1.xaxis.tick_top()
    ax1.set_xticks([-L, 0, L])
    ax1.set_xticklabels([r"$Y = 200\ \textsf{kPa}$",
                         r"$Y = 800\ \textsf{kPa}$",
                         r"$Y = 200\ \textsf{kPa}$"])
    ax1.set_yticks([1.0, 2.0])
    ax1.set_ylabel(r"$\Lambda\ (\textsf{\textmu m})$")

    ax2 = fig.add_subplot(gs[1])
    # ax2.set_xlim([-L, -L + 30])
    # ax2.set_ylim([-L, -L + 30])

    psi = np.load(f"../data/hr_varstiff_{phi_0}.npy")
    ax2.pcolormesh(X, Y, psi, cmap="RdBu", rasterized=True)

    ax2.plot([18, 22], [-22, -22], "w-", linewidth=2.0)

    ax2.set_aspect("equal", adjustable="box")
    ax2.set_box_aspect(1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticks([])

    plt.tight_layout()
    plt.savefig(
        f"varstiff_{phi_0}.pdf",
        crop=True,
        transparent=True,
        optimize=True,
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
