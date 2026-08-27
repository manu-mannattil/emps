# -*- coding: utf-8 -*-
"""Lamella phase-diagram using Maxwell construction.

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

This program reproduces a phase diagram similar to Fig. 1 of Thiele et
al., Phys. Rev. E 87, 042915 (2013) and Fig. 2 of Elder and Grant, Phys.
Rev. E 70, 051605 (2004).
"""

import numpy as np
from scipy.optimize import minimize_scalar

# Parameters in the Landau energy.
a = 1
b = 1
T_c = 0

kappa = 1
h = 1

phi_c = 0.2

# Stretch.
p = 3/2
gamma = 4.0

def params(p, gamma, direction="parallel"):
    """Get nu*kbt, transition temperature, and domain size for given (p, gamma)."""
    # Rubber elasticity.
    M = gamma*kappa/h**2
    nukbt = M/(phi_c**(-5/3) + phi_c**(-1))

    # Phase separation temperature for various phases and various wave
    # vectors q (i.e., parallel or perpendicular to the x axis).
    if direction == "parallel":
        h_eff = h/np.sqrt(p)
        gamma_eff = gamma * 1/p * (1/p + phi_c**(2/3)) / (1 + phi_c**(2/3))
        M_eff = M * (p ** 2 + phi_c**(2/3)) / (1 + phi_c**(2/3))
    else:
        h_eff = h*p
        gamma_eff = gamma * p**2 * (p**2 + phi_c**(2/3)) / (1 + phi_c**(2/3))
        M_eff = M * (1/p + phi_c**(2/3)) / (1 + phi_c**(2/3))

    T = T_c - kappa/a/h_eff**2 * (1 + np.log(gamma_eff))

    return h_eff, M_eff, gamma_eff, T

h_eff, M_eff, gamma_eff, T = params(p, gamma, direction="perpendicular")

# This term appears in the minimized free energies.
Q = kappa / h_eff**2 * (1 + np.log(gamma_eff))

def F(x, adT):
    return adT + 3 * b * x**2 + Q

def f_unif(x, adT):
    return 0.5 * (adT+M_eff) * x**2 + 0.25 * b * x**4

def f_stripe(x, adT):
    return -1 / (6*b) * F(x, adT)**2 + f_unif(x, adT)

def minimize(fun, bounds):
    return minimize_scalar(fun, bounds=bounds, method="bounded")

def stripe_unif(adT):
    x0 = np.sqrt(-(adT + Q) / (3*b))
    # Bounds for the stripe phase
    x1, x2 = 0.1*x0, 0.9 * x0

    # Bounds for the uniform phase
    y1, y2 = 0.9 * x0, 2 * x0

    def fun(mu, find_mu=True):
        f = lambda x: f_stripe(x, adT) - x*mu
        g = lambda y: f_unif(y, adT) - y*mu

        x = minimize(f, bounds=(x1, x2)).x
        y = minimize(g, bounds=(y1, y2)).x

        if find_mu:
            a = f_stripe(x, adT) - f_unif(y, adT)
            b = x - y
            # a, b, and μ should have the same sign.
            if a * b * mu < 0:
                return 1e10
            return abs(a - mu*b)
        else:
            return x, y

    mu = minimize(fun, bounds=(0, 1)).x
    x, y = fun(mu, False)

    # Check if the x, y values are too close to the bounds.
    # If they are, we're still above the tricritical point.
    # In that case, just return the spinodal values.
    thres = 0.00001
    if abs(x1 - x) < thres or abs(x2 - x) < thres or abs(y1 - y) < thres or abs(y2 - y) < thres:
        return [mu, x0, x0]

    return [mu, x, y]

N = 500
r = np.linspace(T - 1e-5, -2.6, N) # adT
r *= a
res = np.array([stripe_unif(_) for _ in r])
x, y = res[:, 1], res[:, 2]

x += phi_c
y += phi_c

# Tricritical point ----------------------------------------------------

# In stripe_unif(), the upper bound for the stripe phase is less than
# the spinodal value.  This is fine well below the tricritical point,
# but close to it, this creates issues and a "kink" is developed near
# the actual tricritical point.  So we start with a guess for the
# temperature of the tricritical point.  This temperature must be
# __below__ the actual point.
rs = -2.1

# Extrapolate the binodals upward and find the intersection point.
r1, x1, y1 = r[r < rs][0], x[r < rs][0], y[r < rs][0]
r2, x2, y2 = r[r < rs][1], x[r < rs][1], y[r < rs][1]
# Intersection point of two lines passing through (x1, r1), (x2, r2) and (y1, r1), (y2, r2).
m1 = (r2-r1) / (x2-x1)
b1 = r1 - m1*x1
m2 = (r2-r1) / (y2-y1)
b2 = r1 - m2*y1
xs, rs = (b2-b1) / (m1-m2), m1 * (b2-b1) / (m1-m2) + b1

print(f"Estimated tricritical point at ({xs:.3}, {rs:.3})")

# Second-order curve.
r_2nd = r[r > rs]
x_2nd = x[r > rs]
r_2nd = np.insert(r_2nd, -1, rs)
x_2nd = np.insert(x_2nd, -1, xs)
r_2nd = np.insert(r_2nd, 0, T) # critical point
x_2nd = np.insert(x_2nd, 0, phi_c) # critical point
# Sort in descending order of the temperature.
i = np.argsort(r_2nd)[::-1]
x_2nd, r_2nd = x_2nd[i], r_2nd[i]

# First-order curves.
r_1st = r[r <= r1]
x_1st = x[r <= r1]
y_1st = y[r <= r1]
r_1st = np.insert(r_1st, 0, rs)
x_1st = np.insert(x_1st, 0, xs)
y_1st = np.insert(y_1st, 0, xs)

# Plotting ------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import charu

rc = {
    "charu.doc": "aps",
    "charu.tex": True,
    "charu.tex.font": "fourier",
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()

    ax.plot(x_2nd, r_2nd, "k--")
    ax.plot(-x_2nd + 2*phi_c, r_2nd, "k--")

    shade = {"color": "C0", "alpha": 0.15}

    ax.plot(x_1st, r_1st, "k")
    ax.plot(y_1st, r_1st, "k")
    ax.fill_betweenx(r_1st, x_1st, y_1st, **shade)
    ax.plot(-x_1st + 2*phi_c, r_1st, "k")
    ax.plot(-y_1st + 2*phi_c, r_1st, "k")
    ax.fill_betweenx(r_1st, -x_1st + 2*phi_c, -y_1st + 2*phi_c, **shade)

    # Tricritical point.
    ax.scatter(xs, rs, color="k", s=15, zorder=100, facecolor="w", linewidth=0.75)
    ax.scatter(-xs + 2*phi_c, rs, color="k", s=15, zorder=100, facecolor="w", linewidth=0.75)

    ax.set_ylim(-2.6, -1.6)
    ax.set_xlim(0.2 - 0.63, 0.2 + 0.63)

    ax.scatter([0.2], [-2.50], s=15, facecolor="k")
    ax.text(0.14, -2.515, "c", horizontalalignment="left")

    ax.scatter([0.67], [-2.50], s=15, facecolor="k")
    ax.text(0.61, -2.515, "d", horizontalalignment="left")

    ax.text(0.2, -2.35, r"L$_{\perp}$", horizontalalignment="center")
    ax.text(0.585, -2.35, r"L$_{\perp}$ + U", horizontalalignment="center")
    ax.text(-0.18, -2.35, r"U + L$_{\perp}$", horizontalalignment="center")
    ax.text(0.7, -1.9, "U", horizontalalignment="center")
    ax.text(-0.3, -1.9, "U", horizontalalignment="center")

    ax.set_xlabel(r"$\phi_0$")
    ax.set_ylabel(r"$T$", rotation=None, va="center", ha="center", labelpad=8)

    plt.tight_layout()
    plt.savefig(
        "lamella_perpendicular_inc.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
