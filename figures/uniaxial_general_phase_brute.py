#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Same as Fig. 3 of Paper II, but the phase boundaries are obtained
# using brute force.  Only for checking.
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw as L

def F_a(x, y):
    return x**-2/y*(1 + np.log(x**4 * y))

def F_b(x, y):
    return x/y*(1 + np.log(x**-2 * y))

def F_ua(x, y):
    return x**2

def F_ub(x, y):
    return x**-1

x, y = 0.490, 8.08
print(F_b(x, y) - F_ua(x, y))

# ----------------------------------------------------------------------

x1 = 0.657829
x2 = 1.52015
dx = 0.35
N = 200

fig, ax = plt.subplots()

x = np.linspace(x1, x2, N)
y = np.exp(2 * np.log(x) * (x**3 + 2)/(x**3 - 1) - 1)
ax.plot(x, y, color="C0", label="A vs. B")

x = np.linspace(x1 - dx, x2 + 2*dx, N)
y = 1/x**4
ax.plot(x, y, color="C1", label=r"$y = 1/x^4$")

x = np.linspace(x1 - dx, x2 + 2*dx, N)
y = x**2
ax.plot(x, y, color="C2", label=r"$y = x^2$")

ax.plot([1, 1], [1, 10], color="C0", label="A vs. B")

ax.set_ylim(0, 10)
ax.set_xlim(x1 - dx, x2 + 2*dx)

x = np.linspace(x1 - dx, 1, N)
y1 = -L(-x**3/np.e, 0)/x
y2 = -L(-x**3/np.e, -1)/x
y1[-1] = y2[-1] = 1
ax.plot(x, y1.real, color="C6", label="lambert")
ax.plot(x, y2.real, color="C6")

x = np.linspace(1, x2 + 2*dx, N)
y1 = -L(-x**-3/np.e, 0)/x
y2 = -L(-x**-3/np.e, -1)/x
y1[0] = y2[0] = 1
ax.plot(x, y1.real, color="C6")
ax.plot(x, y2.real, color="C6")

plt.legend()
plt.show()
