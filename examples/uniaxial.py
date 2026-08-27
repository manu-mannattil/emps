# -*- coding: utf-8 -*-
"""General equilibrium profiles in 2D.

This program produces equilibrium profiles to check the validity of the
general phase diagram at phi = phi_c and just below the phase transition
temperature.
"""

import numpy as np

from pfmodel import *
from utils import *

# Grid points.
n = 256

# Critical parameters.
T_c, phi_c = 0, 0.2

# Ginzburg-Landau energy parameters.
a = 1
b = 1
kappa = 1

# Coarse-graining length scale.
h = 1

def params(p, gamma, phase="lamella", direction="parallel"):
    """Get nu*kbt, transition temperature, and domain size for given (p, gamma)."""
    # Rubber elasticity.
    M = gamma*kappa/h**2
    nukbt = M/(phi_c**(-5/3) + phi_c**(-1))

    # Phase separation temperature for various phases and various wave
    # vectors q. The direction determines if the lamella (or growth) are
    # parallel or perpendicular to the x axis.  Note that for parallel
    # lamella, q is perpendicular to the x axis, and vice versa.
    if direction == "parallel":
        h_eff = h/np.sqrt(p)
        gamma_eff = gamma * 1/p * (1/p + phi_c**(2/3)) / (1 + phi_c**(2/3))
    else:
        h_eff = h*p
        gamma_eff = gamma * p**2 * (p**2 + phi_c**(2/3)) / (1 + phi_c**(2/3))

    if phase == "lamella":
        T = T_c - kappa/a/h_eff**2 * (1 + np.log(gamma_eff))
        domain_size = 2*np.pi*h_eff/np.sqrt(np.log(gamma_eff))
    else:
        domain_size = 10*h_eff
        T = T_c - kappa/a/h_eff**2 * gamma_eff

    return nukbt, T, domain_size

for p, gamma, name in [[0.64, 2.70, "perpendicular_growth"],
                       [0.84, 3.18, "parallel_lamella"],
                       [0.91, 1.77, "perpendicular_lamella"],
                       [1.10, 1.54, "parallel_lamella"],
                       [1.36, 3.10, "perpendicular_lamella"],
                       [1.50, 1.45, "parallel_growth"]]:
    direction, phase = name.split("_")
    nukbt, T, domain_size = params(p, gamma, phase=phase, direction=direction)

    # Go slightly below the phase transition temperature,
    T -= 0.02
    # Choose a box length that is appropriate.
    L = 5 * domain_size

    fname = f"../data/uniaxial_{name}_{p}_{gamma}.npy"
    #psi_0 = np.load(fname)

    if phase == "growth":
        max_steps = 100
        dt = 1
    else:
        max_steps = 10*1000
        dt = 5

    pfm = PFUniaxial(T=T,
                     phi_0=phi_c,
                     a=a,
                     b=b,
                     T_c=T_c,
                     phi_c=phi_c,
                     kappa=kappa,
                     nukbt=nukbt,
                     p=p,
                     h=h,
                     L=L,
                     n=n,
                     dt=dt,
                     disorder=0.01)

    for i in range(max_steps):
        pfm.evolve()

    fname = f"../data/uniaxial_{name}_{p}_{gamma}.npy"
    np.save(fname, pfm.psi)
    print(f"p = {p}; gamma = {gamma} done!")

for p, gamma, name in [[3/2, 4, "offcrit_0.67"],
                       [3/2, 4, "crit_0.2"],
                       [2/3, 4, "offcrit_0.58"],
                       [2/3, 4, "crit_0.2"]]:
    if p < 1:
        direction = "parallel"
    else:
        direction = "perpendicular"

    phi_0 = float(name.split("_")[1])
    nukbt, _, domain_size = params(p, gamma, phase="lamella", direction=direction)

    # Go slightly below the phase transition temperature,
    T = -2.5
    # Choose a box length that is appropriate.
    L = 4 * domain_size

    max_steps = 10000

    pfm = PFUniaxial(T=T,
                    phi_0=phi_0,
                    a=a,
                    b=b,
                    T_c=T_c,
                    phi_c=phi_c,
                    kappa=kappa,
                    nukbt=nukbt,
                    p=p,
                    h=h,
                    L=L,
                    n=n,
                    d=2,
                    dt=1,
                    disorder=1)

    for i in range(max_steps):
        pfm.evolve()

    fname = f"../data/uniaxial_{name}_{p:.2}_{gamma}.npy"
    np.save(fname, pfm.psi)
    print(f"p = {p:.2}; gamma = {gamma}; phi_0 = {phi_0} done!")

