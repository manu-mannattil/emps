# -*- coding: utf-8 -*-
"""Two-dimensional (general) phase-diagram using Maxwell construction.

This program computes the phase-boundary curves using a common tangent
construction.  The basic algorithm is as follows:

1. For a given temperature [measured as the parameter adT = a(T-T_c)],
   start with some value of the chemical potential μ.

2. Simultaneously solve df/dx = μ and dg/dy = μ for x and y.  Here f(x,
   aΔT) and g(y, aΔT) are two free-energy functions representing two
   different phases, and x and y represent the values of the order
   parameter in the two phases.  Check if the found x and y values
   satisfy the osmotic pressure condition, i.e., μ = [f(x) - g(y)]/(x
   - y).  Most likely it won't.

3. Optimize μ until a suitable triplet (μ, x, y) is found for a given
   temperature.

4. For a new temperature, start with Step 1.

This program reproduces a phase diagram similar to Fig. 10 of Thiele et
al., Phys. Rev. E 87, 042915 (2013) and Fig. 4 of Elder and Grant, Phys.
Rev. E 70, 051605 (2004).
"""

import numpy as np
from scipy.optimize import minimize_scalar

# Parameters
T_c = 0
phi_c = 0.2
kappa, h, M = 1, 1, 4
zeta = M * h**2 / kappa

# This term appears in the minimized free energies.
Q = kappa / h**2 * (1 + np.log(zeta))

# Parameters in the Landau energy.
a = 1
b = 1

def dfun(fun, x, args=(), h=1e-6):
    """Use the fourth-order central difference formula to compute dfun/dx."""
    f1 = fun(x + h, *args)
    f2 = fun(x + 2*h, *args)
    b1 = fun(x - h, *args)
    b2 = fun(x - 2*h, *args)

    return (-f2 + 8*f1 - 8*b1 + b2) / (12*h)

def F(x, adT):
    return adT + 3 * b * x**2 + Q

def f_unif(x, adT):
    return 0.5 * (adT+M) * x**2 + 0.25 * b * x**4

def f_hex(x, adT):
    A = 4 / (15*b) * (3*b*x + np.sqrt(9 * b**2 * x**2 - 15 * b * F(x, adT)))
    return 1 / 64 * A * A * (6 * F(x, adT) - 3*b*x*A) + f_unif(x, adT)

def f_stripe(x, adT):
    return -1 / (6*b) * F(x, adT)**2 + f_unif(x, adT)

def minimize(fun, bounds):
    return minimize_scalar(fun, bounds=bounds, method="bounded")

def hex_unif(adT):
    x0 = np.sqrt(-15 * (adT+Q) / (36*b))
    # Bounds for the hexagonal phase
    x1, x2 = 0, x0

    # Bounds for the uniform phase
    y1, y2 = x0, 1

    def fun(mu, find_mu=True):
        # Either minimize f_hex(x, adT) - μx or solve f_hex'(x, adT) = μ to
        # find x (and y)
        f = lambda x: f_hex(x, adT) - x*mu
        # f = lambda x: abs(dfun(f_hex, x, args=(adT,)) - mu)

        g = lambda y: f_unif(y, adT) - y*mu
        # g = lambda y: abs(dfun(f_unif, y, args=(adT,)) - mu)

        x = minimize(f, bounds=(x1, x2)).x
        y = minimize(g, bounds=(y1, y2)).x

        if find_mu:
            p = f_hex(x, adT) - f_unif(y, adT)
            q = x - y
            # p, q, and μ should have the same sign.
            if p * q * mu < 0:
                return 1e10
            return abs(p - mu*q)
        else:
            return x, y

    mu = minimize(fun, bounds=(0, 1)).x
    x, y = fun(mu, False)
    return [mu, x, y]

def hex_stripe(adT):
    x0 = np.sqrt(-15 * (adT+Q) / (36*b))
    # Bounds for the hexagonal phase
    x1, x2 = 0, x0

    # Bounds for the stripe phase
    y1, y2 = 0, x0 / 2

    def fun(mu, find_mu=True):
        # Either minimize f_hex(x, adT) - μx or solve f_hex'(x, adT) = μ to
        # find x (and y)
        f = lambda x: f_hex(x, adT) - x*mu
        # f = lambda x: abs(dfun(f_hex, x, args=(adT,)) - mu)

        g = lambda y: f_stripe(y, adT) - y*mu
        # g = lambda y: abs(dfun(f_unif, y, args=(adT,)) - mu)

        x = minimize(f, bounds=(x1, x2)).x
        y = minimize(g, bounds=(y1, y2)).x

        if find_mu:
            p = f_hex(x, adT) - f_stripe(y, adT)
            q = x - y
            # p, q, and μ should have the same sign.
            if p * q * mu < 0:
                return 1e10
            return abs(p - mu*q)
        else:
            return x, y

    mu = minimize(fun, bounds=(0, 1)).x
    x, y = fun(mu, False)
    p = f_hex(x, adT) - f_stripe(y, adT)
    q = x - y
    return [mu, x, y]

# Plotting -------------------------------------------------------------

import matplotlib.pyplot as plt
import charu

rc = {
    "charu.doc": "aps",
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "figure.figsize": [260 * charu.pt, 260 / charu.golden * charu.pt],
}

with plt.rc_context(rc):
    # Number of points along each coexistence curve in the phase diagram.
    N = 200

    # Start from a T slightly below the temperature where you'd expect microphase separation.
    T = np.linspace(T_c - Q/a - 1e-5, -3.5, N)
    mark = -10
    adT = a * (T-T_c)

    fig, ax = plt.subplots()

    shade = {"color": "#d8dee9ff"}

    res1 = np.array([hex_stripe(_) for _ in adT])
    res1 += phi_c

    ax.plot(res1[:, 1], T, "#434c5e")
    ax.plot(res1[:, 2], T, "#434c5e")
    ax.fill_betweenx(T, res1[:, 1], res1[:, 2], **shade)

    ax.plot(2*phi_c - res1[:, 1], T, "#434c5e")
    ax.plot(2*phi_c - res1[:, 2], T, "#434c5e")
    ax.fill_betweenx(T, 2*phi_c - res1[:, 1], 2*phi_c - res1[:, 2], **shade)

    res2 = np.array([hex_unif(_) for _ in adT])
    res2 += phi_c

    ax.plot(res2[:, 1], T, "#434c5e")
    ax.plot(res2[:, 2], T, "#434c5e")
    ax.fill_betweenx(T, res2[:, 1], res2[:, 2], **shade)

    ax.plot(2*phi_c -res2[:, 1], T, "#434c5e")
    ax.plot(2*phi_c -res2[:, 2], T, "#434c5e")
    ax.fill_betweenx(T, 2*phi_c - res2[:, 1], 2*phi_c - res2[:, 2], **shade)

    # Mark points on the phase diagram.
    T_mark = -3.4
    res3 = hex_stripe(a * (T_mark-T_c))
    res4 = hex_unif(a * (T_mark-T_c))
    psi_mark = np.asarray([0.5 * (res3[1] + res4[1]), 0.5 * (res4[1] + res4[2])])
    ax.scatter(phi_c + psi_mark[0], T_mark, color="black", s=10, marker="x")
    ax.scatter(phi_c + psi_mark[1], T_mark, color="black", s=10, marker="x")

    # ax.text(phi_c + psi_mark[0], -3.36, "b", horizontalalignment="center")
    # ax.text(phi_c + psi_mark[1], -3.36, "c", horizontalalignment="center")
    # ax.text(phi_c + psi_mark[2], -3.36, "d", horizontalalignment="center")
    # ax.text(phi_c + psi_mark[3], -3.36, "e", horizontalalignment="center")

    # Put phase labels.
    T_mark = -3.2
    res3 = hex_stripe(a * (T_mark-T_c))
    res4 = hex_unif(a * (T_mark-T_c))
    psi_mark = [0.5 * (res3[1] + res3[2]) - 0.015, 0.5 * (res4[1] + res4[2]) - 0.035]
    rotation = [-85, -71]
    for i, s in enumerate([r"L + H", r"H + U"]):
        ax.text(phi_c + psi_mark[i], T_mark, s, horizontalalignment="center", rotation=rotation[i])
    for i, s in enumerate([r"IH + L", r"U + IH"]):
        ax.text(phi_c - psi_mark[i], T_mark, s, horizontalalignment="center", rotation=-rotation[i])
    ax.text(phi_c + 0.5, T[0] - 0.2, "U", horizontalalignment="center")
    ax.text(phi_c -0.5, T[0] - 0.2, "U", horizontalalignment="center")

    psi_mark = [-0.5 * (res3[1] + res4[1]), 0, 0.5 * (res3[1] + res4[1])]
    for i, s in enumerate([r"IH", r"L", r"H"]):
        ax.text(phi_c + psi_mark[i], T_mark + 0.03, s, horizontalalignment="center")

    # Label critical point.
    # ax.plot([-3, 3], [T[0], T[0]], "--", color="#aaaaaa", zorder=-100)
    ax.scatter(phi_c, T[0], color="black", s=5, zorder=100)
    ax.text(phi_c, T[0] + 0.05, r"$T_{c}'$", horizontalalignment="center")

    ax.set_xlim(phi_c - 0.8, 1.0)
    ax.set_ylim(-3.5, -2.25)
    ax.set_xlabel(r"$\phi_0$", labelpad=6)
    ax.set_ylabel(r"$T$", va="center", ha="center", labelpad=8)

    # Linear stability curves for hexagons and stripes.
    # ax.plot(np.sqrt(-15 * (adT+Q) / (36*b)), T, "r--") # hexagons
    # ax.plot(np.sqrt(-1 * (adT+Q) / (3*b)), T, "b--") # stripes

    plt.tight_layout()
    plt.savefig(
        "isotropic_general_phase.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
