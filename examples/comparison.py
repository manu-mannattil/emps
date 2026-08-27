# -*- coding: utf-8 -*-
"""Elastomer microphase separation in 2D."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from params import parameters
from pfmodel import PFUniaxial
from utils import *

Y = 350
kwargs = parameters(Y, kwargs=True)

L = kwargs["L"]
n = 512
T = 20

max_steps = 20*100
for phi_0 in [0.2, 0.5]:
    if phi_0 == 0.2:
        p_list = np.linspace(0.5, 1, 4)[::-1]
    else:
        p_list = np.linspace(1, 2, 4)

    for num, p in enumerate(p_list):
        pfm = PFUniaxial(T=T, p=p, phi_0=phi_0, n=n, **kwargs, dt=1, disorder=0.1)

        for i in range(max_steps):
            pfm.evolve()

        fname = f"../data/comparison_{phi_0}_{num}.npy"
        np.save(fname, pfm.psi)
        print(f"comparison phi_0 = {phi_0}, num = {num} done!")
