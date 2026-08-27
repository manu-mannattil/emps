#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# - The variable x represents the anisotropy factor p.
# - The variable y represents the (inverse) elastocapillary number.

import numpy as np
from scipy.special import lambertw as L
import matplotlib.pyplot as plt
import charu

# Critical volume fraction:
# only phi^(2/3) appears in all equations.
phi = 0.2
phi = phi**(2/3)

def F_lam_par(x, y):
    """Parallel lamella."""
    gamma = x**2 * y * (x**2 + phi)/(1 + phi)
    if gamma < 1:
        return np.inf
    return x**-2*(1 + np.log(gamma))

def F_lam_per(x, y):
    """Perpendicular lamella."""
    gamma = 1/x * y * (1/x + phi)/(1 + phi)
    if gamma < 1:
        return np.inf
    return x*(1 + np.log(gamma))

def F_two_par(x, y):
    """Parallel two phase."""
    return y*(x**2 + phi)/(1 + phi)

def F_two_per(x, y):
    """Perpendicular two phase."""
    return y*(1/x + phi)/(1 + phi)

def color(x, y):
    """Return color based on phase."""
    lam_par = F_lam_par(x, y)
    lam_per = F_lam_per(x, y)
    two_par = F_two_par(x, y)
    two_per = F_two_per(x, y)
    m = min([lam_par, lam_per, two_par, two_per])

    if m == lam_par:
        return [0.8, 0.8, 1]    # blue
    elif m == lam_per:
        return [1, 0.8, 0.8]    # red
    elif m == two_par:
        return [0.8, 1.0, 0.8]  # light green
    else:
        return [1.0, 1.0, 0.8]    # light yellow

# These are your intersection points obtained using Mathematica's
# FindRoot[].
x_left, x_right = 0.719, 1.42
y_left, y_right = 3.03, 1.82

x_min, x_max = x_left - 0.2, x_right + 0.2
y_min, y_max = 0.75, 4

rc = {
    "charu.doc": "aps",
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "figure.figsize": [260 * charu.pt, 268 / charu.golden * charu.pt],
}

with plt.rc_context(rc):
    # Number of grid points.
    N = 500
    fig, ax = plt.subplots()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    parallel = {"color": "C1", "alpha": 0.15}
    perpendicular = {"color": "C0", "alpha": 0.15}

    # Brute force phase diagram --------------------------------------------

    # Here we just compare the values of the different effective binary
    # interactions and see which one is the lowest.  That's the phase that
    # appears first during a temperature quench.

    # x = np.linspace(x_min, x_max, N)
    # y = np.linspace(y_min, y_max, N)
    # X, Y = np.meshgrid(x, y)

    # im = np.empty((N, N, 3))
    # for i in np.ndindex(X.shape):
    #     im[i] = color(X[i], Y[i])

    # ax.pcolormesh(X, Y, im)

    # Analytical phase boundaries ------------------------------------------

    # Parallel lamella vs. perpendicular lamella.
    x = np.linspace(x_left, 1, N)
    A = x**2*(x**2 + phi)/(1 + phi)
    B = 1/x*(1/x + phi)/(1 + phi)
    y1 = (x**-2*(1 + np.log(A)) - x*(1 + np.log(B)))/(x - x**-2)
    y1 = np.exp(y1)
    ax.plot(x, y1, "k--")
    x1_top = x
    y1_top = y1

    # Parallel two-phase vs. parallel lamella.
    A = x**2*(x**2 + phi)/(1 + phi)
    y2 = 1/A
    ax.plot(x, y2, "k-")

    ax.fill_between(x, y1, y2, **perpendicular)

    # Parallel lamella vs. perpendicular lamella.
    x = np.linspace(1, x_right, N)
    A = x**2*(x**2 + phi)/(1 + phi)
    B = 1/x*(1/x + phi)/(1 + phi)
    y1 = (x**-2*(1 + np.log(A)) - x*(1 + np.log(B)))/(x - x**-2)
    y1 = np.exp(y1)
    ax.plot(x, y1, "k--")
    x2_top = x
    y2_top = y1

    # Perpendicular two-phase vs. perpendicular lamella.
    x = np.linspace(1, x_right, N)
    B = 1/x*(1/x + phi)/(1 + phi)
    y2 = 1/B
    ax.plot(x, y2, "k-")

    ax.fill_between(x, y1, y2, **parallel)

    # Perpendicular two-phase vs. parallel lamella.
    x = np.linspace(x_right, x_max)
    A = x**2*(x**2 + phi)/(1 + phi)
    B = 1/x*(1/x + phi)/(1 + phi)
    y = -L(-B*x**3/(A*np.e), -1)/(B*x**3)
    ax.plot(x, y, "k-")

    x2_top = np.hstack((x2_top, x))
    y2_top = np.hstack((y2_top, y))
    ax.fill_between(x2_top, y2_top, y_max, **perpendicular)

    # Perpendicular lamella vs. parallel two-phase.
    x = np.linspace(x_min, x_left)
    A = x**2*(x**2 + phi)/(1 + phi)
    B = 1/x*(1/x + phi)/(1 + phi)
    y = -L(-A/(B*x**3*np.e), -1)*x**3/A
    ax.plot(x, y, "k-")

    x1_top = np.hstack((x, x1_top))
    y1_top = np.hstack((y, y1_top))
    ax.fill_between(x1_top, y1_top, y_max, **parallel)

    # Vertical dashed line.
    ax.plot([1, 1], [1, y_max], "k--")
    ax.plot([1, 1], [0.75, 1], "--", color="#cccccc", zorder=-100)

    ax.scatter([0.64], [2.70], s=15, facecolor="k")
    ax.text(0.64, 2.45, "d", horizontalalignment="center")
    ax.text(0.7, 1.45, r"two phase", horizontalalignment="center")

    ax.scatter([0.84], [3.18], s=15, facecolor="k")
    ax.text(0.84, 2.93, "b", horizontalalignment="center")
    ax.text(0.82, 3.70, r"parallel lamellae (L$_\|$)", horizontalalignment="center")
    ax.text(0.82, 3.45, r"$q_x, q_y \neq 0; q_z = 0$", horizontalalignment="center")

    ax.scatter([0.91], [1.77], s=15, facecolor="k")
    ax.text(0.91, 1.93, "e", horizontalalignment="center")

    ax.scatter([1.10], [1.54], s=15, facecolor="k")
    ax.text(1.1, 1.7, "f", horizontalalignment="center")

    ax.text(1.16, 1.56, r"L$_\|$")
    ax.text(0.82, 2.23, r"L$_\perp$")

    ax.scatter([1.36], [3.10], s=15, facecolor="k")
    ax.text(1.36, 2.85, "c", horizontalalignment="center")
    ax.text(1.36, 3.62, r"perpendicular lamellae (L$_\perp$)", horizontalalignment="center")
    ax.text(1.36, 3.37, r"$q_x = q_y = 0; q_z \neq 0$", horizontalalignment="center")

    ax.scatter([1.50], [1.45], s=15, facecolor="k")
    ax.text(1.5, 1.2, "g", horizontalalignment="center")
    ax.text(1.35, 1.2, r"two phase", horizontalalignment="center")

    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$\gamma_0$", rotation=None, va="center", ha="center", labelpad=8)

    plt.tight_layout()
    plt.savefig(
        "uniaxial_general_phase_inc.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
