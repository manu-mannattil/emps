# -*- coding: utf-8 -*-
"""Two-dimensional phase-diagram using Maxwell construction.

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
"""

import numpy as np
from scipy.optimize import minimize_scalar
from params import parameters

# Experimental values of T_micro from the paper (in celsius) and phi (estimated).
# Y = 800 # Young's modulus in kPa
# T_exp = np.array([32.70033223, 23.72624585, 15.02059801, 6.52225914, -7.37940199])
# phi_exp = np.array([0.44148944, 0.5217842, 0.59499301, 0.64427642, 0.71333633])

Y = 350
T_exp = np.array([52.00473347, 46.7140014, 36.99424453, 27.4928729, 12.50282395])
phi_exp = np.array([0.40804667, 0.47110599, 0.5447087, 0.61854361, 0.68804275])

# Choose phi to be the mean of phi_exp for computing kappa, h, and M.
# This is mostly an estimate based on Fig. S7 of the Nat. Mat. paper.
T_c = 70
phi_c = 0.2
kappa, h, nukbt, zeta = parameters(Y)
M = nukbt/phi_c**2 * (phi_c + phi_c**(1/3))

# This term appears in the minimized free energies.
Q = kappa / h**2 * (1 + np.log(zeta))

# Parameters in the Landau energy (estimated).
a = 2.5e-2 # kPa/K
b = 2 # kPa

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
    y1, y2 = x0 / 2, 1

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

    mu = minimize(fun, bounds=(0, 5000)).x
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

    mu = minimize(fun, bounds=(0, 5000)).x
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
}

with plt.rc_context(rc):
    # Number of points along each coexistence curve in the phase diagram.
    N = 100

    # Start from a T slightly below the temperature where
    T = np.linspace(T_c - Q/a - 1e-5, -10, N)
    adT = a * (T-T_c)

    fig, ax = plt.subplots()

    shade1 = {"color": "C1", "alpha": 0.15}
    shade2 = {"color": "C0", "alpha": 0.15}

    res1 = np.array([hex_stripe(_) for _ in adT])
    ax.plot(phi_c + res1[:, 1], T, "C0")
    ax.plot(phi_c - res1[:, 1], T, "C0")
    ax.fill_betweenx(T, phi_c + res1[:, 1], phi_c - res1[:, 1], **shade1)

    res2 = np.array([hex_unif(_) for _ in adT])
    ax.plot(phi_c + res2[:, 1], T, "C0")
    ax.plot(phi_c - res2[:, 1], T, "C0")
    ax.fill_betweenx(T, phi_c + res1[:, 1], phi_c + res2[:, 1], **shade2)
    ax.fill_betweenx(T, phi_c - res1[:, 1], phi_c - res2[:, 1], **shade2)

    # The stripe phase does not exist beyond this curve.
    b_stripe = np.sqrt(-(adT + Q) / (3*b))
    ax.plot(phi_c + b_stripe, T, "--", color="#333333", zorder=10)
    ax.plot(phi_c - b_stripe, T, "--", color="#333333", zorder=10)

    # The hexagonal phase does not exist beyond this curve (not plotted).
    # b_stripe = np.sqrt(-15 * (adT + Q) / (36*b))
    # ax.plot(phi_c + b_stripe, T, "r--", zorder=10)
    # ax.plot(phi_c - b_stripe, T, "r--", zorder=10)

    ax.scatter(phi_exp, T_exp, s=43, facecolor="w", zorder=10)
    ax.scatter(phi_exp, T_exp, s=15, facecolor="none", edgecolor="k", linewidth=0.75, zorder=100)

    ax.set_xlim(0, 0.8)
    ax.set_ylim(-10, 75)
    ax.set_xlabel(r"$\phi_0$")
    ax.set_ylabel(r"$T\ (^\circ\!\mathrm{C})$")

    # ax.scatter([0.2], [20], s=15, facecolor="none", edgecolor="k", linewidth=0.75, zorder=100)
    # ax.scatter([0.42], [20], s=15, facecolor="none", edgecolor="k", linewidth=0.75, zorder=100)

    ax.scatter([0.2], [T_c - Q/a], s=15, facecolor="k", zorder=200)
    ax.text(0.2, T_c - Q/a + 5, r"$T_{\mathrm{c}}'$", horizontalalignment="center")

    ax.text(0.2, 12, "b", horizontalalignment="center")
    ax.scatter([0.2], [20], s=15, facecolor="k")

    ax.text(0.5, 12, "c", horizontalalignment="center")
    ax.scatter([0.5], [20], s=15, facecolor="k")

    ax.text(0.2, 0, "lamellae", horizontalalignment="center")
    ax.text(0.55, 0, "hexagonal phase", horizontalalignment="center")
    ax.text(0.65, 60, "uniform", horizontalalignment="center")

    plt.tight_layout()
    plt.savefig(
        "uniaxial_real_inc.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        bbox_inches="tight",
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
