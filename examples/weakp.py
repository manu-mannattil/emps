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

# Rubber elasticity.
gamma = 4.0
M = gamma*kappa/h**2
nukbt = M/(phi_c**(-5/3) + phi_c**(-1))
domain_size = 2*np.pi*h/np.sqrt(np.log(gamma))
print(domain_size)
exit()

# Temperature.
T = -3.4

max_steps = 20*1000
for phi_0 in [0.61, 0.83]:
    for num, p in enumerate(np.linspace(1.0, 1.5, 7)):
        L = 7 * domain_size

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
                         dt=2,
                         disorder=1)

        for i in range(max_steps):
            pfm.evolve()

        fname = f"../data/weakp_{phi_0}_{num}.npy"
        np.save(fname, pfm.psi)
        print(f"phi_0 = {phi_0}, num = {num} done!")
