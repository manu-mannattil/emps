import numpy as np
from pfmodel import *
from utils import *

T_c = 70 # critical temperatue
phi_c = 0.2 # critical volume fraction
B = 0.024 # parameter B (kPa um^2; used in calculating end-to-end distance)

T = 20 # temperature (in Celsius)
phi_0 = phi_c + 0.3 # network volume fraction

n = 512 # number of grid points along a dimension
L = 30 # length of box (in um)
h_num = 3 # number of constant h field used

# Stiffness field.
x = np.linspace(-L, L, n)
X, _ = np.meshgrid(x, x)
Y_min, Y_max = 200, 800 # dry elastomer stiffness range.
sigma = 15
C = (Y_max-Y_min) / (1 - np.exp(-L**2 / sigma**2))
Y_field = C * np.exp(-X**2 / sigma**2) + (Y_max-C)

# h and M fields.
h_field = 35 * np.sqrt(3 * B / Y_field) * phi_c**(-1 / 3)
M_field = phi_c**(-5 / 3) * Y_field / 3

# psi_0 = np.load(f"../data/varstiff_{phi_0:.2}.npy")
# psi_0 = resize(psi_0, n)
# psi_0 += 0.01*(0.5 - np.random.random((n, n)))

psi_0 = None

pfm = PFVarStiff(T=T,
                 T_c=T_c,
                 phi_c=phi_c,
                 psi_0=psi_0,
                 phi_0=phi_0,
                 h_field=h_field,
                 M_field=M_field,
                 L=L,
                 n=n,
                 h_num=h_num,
                 dt=0.0002*2,
                 disorder=0.6)

# max_steps = 2 * 1000 * 1000

# for i in range(max_steps):
#     pfm.evolve()
#     if i % 100 == 0:
#         print(f"phi_0 = {phi_0}; step = {i}")
    
#     if i > 100*1000 and i % 100*1000 == 0:
#         np.save(f"../data/LargeL_varstiff_{phi_0:.2}_{i}.npy", pfm.psi)

# np.save(f"../data/LargeL_varstiff_{phi_0:.2}.npy", pfm.psi)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = np.linspace(-L, L, n)
X, Y = np.meshgrid(x, x)
fig, ax = plt.subplots()
im = ax.pcolormesh(X, Y, rescale(pfm.psi), cmap="RdBu")
ax.set_aspect("equal")

def animate(i):
    pfm.evolve()
    im.set_array(rescale(pfm.psi))
    return [im]

anim = FuncAnimation(fig, animate, frames=1, interval=1, blit=True)
plt.show()
