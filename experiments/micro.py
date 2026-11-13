# -*- coding: utf-8 -*-

"""Microphase separation temperature as a function of φ0 for various Y.""" 

import numpy as np

Y_list = np.array([800, 350, 180, 40, 10])

# Density of PDMS and HFBMA (kg/m^3)
rho_PDMS = 970
rho_HFBMA = 1345

@np.vectorize
def c_to_phi(c):
    return (1 - c)*rho_HFBMA/((1 - c)*rho_HFBMA + c*rho_PDMS)

micro = np.loadtxt("figs7.dat")

# From the experimental data, extract swelling and microphase
# separation, temperatures for each Y for different swelling
# temperatures.  For example, T_micro[0] contains microphase seperation
# temperatures for T = 80, 60, 50, 40, and 23, for Y = 800.
T_swell = np.asarray([micro[:5][:, 2],
                      micro[5:10][:, 2],
                      micro[10:15][:, 2],
                      micro[15:20][:, 2],
                      micro[20:][:, 2]])
T_micro = np.asarray([micro[:5][:, 4],
                      micro[5:10][:, 4],
                      micro[10:15][:, 4],
                      micro[15:20][:, 4],
                      micro[20:][:, 4]])

c_swell = np.asarray([micro[:5][:, 1],
                      micro[5:10][:, 1],
                      micro[10:15][:, 1],
                      micro[15:20][:, 1],
                      micro[20:][:, 1]])
c_micro = np.asarray([micro[:5][:, 3],
                      micro[5:10][:, 3],
                      micro[10:15][:, 3],
                      micro[15:20][:, 3],
                      micro[20:][:, 3]])

# In the experiments, c_micro is _assumed_ to be the same as c_swell.
# Thus, average the two, convert to a fraction, and then estimate phi,
# the volume fraction of the PDMS network.
c_avg = 0.5*(c_swell + c_micro)
c_avg /= 100
phi = c_to_phi(c_avg)

print("phi_0 (max) = ", phi.max())
print("phi_0 (min) = ", phi.min())
print("phi_0 (avg) = ", phi.mean())

data = []
for i, Y in enumerate(Y_list):
    merge = []
    merge += [np.repeat(Y, len(phi[i]))]
    merge += [phi[i]]
    merge += [T_micro[i]]
    merge = np.vstack(merge)
    data += [merge.T]

data = np.array(data)
np.save("micro.npy", data)
