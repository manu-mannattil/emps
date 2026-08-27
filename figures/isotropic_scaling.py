# -*- coding: utf-8 -*-

import numpy as np
from params import parameters

data = np.load("../experiments/micro.npy")

# Parameters in the Landau energy (estimated).
a = 0.025 # kPa/K
b = 2 # kPa

# This is mostly an estimate based on Fig. S7 of the Nat. Mat. paper.
T_c = 70 # in Celsius
phi_c = 0.2

# Other parameters (affine model)
# B = 0.024 # kPa um^2
# n = 35 # number of cross-links we coarse-grain over.

# Other parameters (phantom model)
B = 0.024 # kPa um^2
n = 35 # number of cross-links we coarse-grain over.

# Interface parameter.
kappa = 0.013 # kPa um^2

def q_theory(Y):
    """Domain size (in μm) as a function of Y (in kPa)."""
    # Eq. (13), step by step (phantom)
    q2 = Y.copy()
    q2 *= np.log(B * n**2 / kappa * (phi_c**(-5 / 3) + phi_c**(-7 / 3)))
    q2 *= phi_c**(2 / 3)
    q2 /= 3 * B * n**2

    return np.sqrt(q2)

def T_m(Y, phi):
    """Microphase separation temperature (in C) as a function of Y (in kPa)."""
    kappa, h, _, zeta = parameters(Y)

    # This term appears in the minimized free energies.
    Q = kappa / h**2 * (1 + np.log(zeta))

    return T_c - (3 * b * (phi - phi_c)**2 + Q) / a

# Plotting -------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.ticker
import charu

rc = {
    "charu.doc": "aps",
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "figure.figsize": [520 * charu.pt, 300 * charu.pt / charu.golden],
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
}

with plt.rc_context(rc):
    fig, axes = plt.subplots(1, 3)

    labelpos = (0.1, 0.9)

    # Domain size ----------------------------------------------------------

    ax = axes[0]
    ax.set_box_aspect(1)

    Y_exp, size, err = np.loadtxt("../experiments/size.dat", usecols=(0, 1, 3), unpack=True)
    err = err / 2

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$Y$ (kPa)")
    ax.set_ylabel(r"$q_{\mathrm{m}}$ (\textmu m$^{-1}$)", labelpad=8)
    ax.set_xlim(8, 1000)
    ax.set_xticks([10, 40, 80, 180, 350, 800])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([1, 2, 4])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Experimental plot.
    q = 2 * np.pi / size
    ax.plot(Y_exp, q, "C0o", markerfacecolor="none", zorder=100)

    # Theoretical plot.
    Y_th = np.linspace(8, 1000, 100)
    ax.plot(Y_th, q_theory(Y_th), "--", color="#999999")

    ax.plot(Y_th[1:30], 1.45 * q_theory(Y_th[1:30]), ":", color="black", linewidth=0.5)
    ax.text(0.32,
            0.55,
            r"$\sim Y^{\small 1/2}$",
            transform=ax.transAxes,
            color="black",
            size=7,
            rotation=45)

    ax.text(*labelpos, r"\textbf{(a)}", transform=ax.transAxes, fontsize="large", ha="center")

    # Microphase separation temperature I ----------------------------------  

    ax = axes[1]
    ax.set_box_aspect(1)

    T_swell = [80, 60, 50, 40, 23]

    # Guideline intercepts
    slope = [-0.045, -0.045, -0.045, -0.045, -0.045]
    intercepts = [69.5, 62.35, 51.81, 39., 23.1]

    # These are T_m and phi_0 for an initial swelling temperature of 60 C.
    for i in range(5):
        color = f"C{i}"

        Y_list = data[:, i][:, 0]
        Y_pad = np.hstack([[1000], Y_list, [-100]])
        phi_exp = data[:, i][:, 1]
        T_m_exp = data[:, i][:, 2]

        # Theoretical microphase temperature.
        T_m_th = []
        for j, Y in enumerate(Y_list):
            kappa, h, nukbt, zeta = parameters(Y)
            Q = kappa / h**2 * (1 + np.log(zeta))
            T_m_th += [T_c - (3 * b * (phi_exp[j] - phi_c)**2 + Q) / a]

        # Make the coefficient matrix; note the transpose.
        A = np.vstack([Y_list, np.ones(len(Y_list))]).T
        m, c = np.linalg.lstsq(A, T_m_th)[0]
        ax.plot(Y_pad, slope[i]*Y_pad + intercepts[i], "--", color=color)

        ax.scatter(Y_list, T_m_exp, s=43, facecolor="w", zorder=20)
        ax.scatter(Y_list,
                   T_m_exp,
                   s=15,
                   facecolor="none",
                   edgecolor=color,
                   linewidth=0.75,
                   zorder=100)
        ax.plot(Y_list, T_m_th, "x", zorder=100, color=color)

    legend_source = []
    for i in range(5):
        # Just for the legend.
        legend_source.append(ax.plot([], [],
                "--o",
                markerfacecolor="w",
                color=f"C{i}",
                label=r"$T_\mathrm{s}$" + f" = {T_swell[i]}" + r" ${}^{\circ}\mathrm{C}$")[0])
    
    legend_top = ax.legend(handles=legend_source[:3])
    ax.add_artist(legend_top)
    ax.legend(handles=legend_source[3:], loc=(0.18, 0.025))

    ax.set_xlabel("$Y$ (kPa)")
    ax.set_ylabel(r"$T_\mathrm{m}$ (${}^{\circ}\mathrm{C}$)")
    ax.set_xlim(-50, 850)
    ax.text(0.1, 0.1, r"\textbf{(b)}", transform=ax.transAxes, fontsize=9, ha="center")

    # Microphase separation temperature II ---------------------------------  

    ax = axes[2]
    ax.set_box_aspect(1)

    for i in range(4, -1, -1):
        Y, phi, T = data[i].T

        color = f"C{4 - i}"

        ax.scatter(phi, T, s=43, facecolor="w", zorder=10)
        ax.scatter(phi, T, s=15, facecolor="none", edgecolor=color, linewidth=0.75, zorder=100)

        kappa, h, nukbt, zeta = parameters(Y[0])
        Q = kappa / h**2 * (1 + np.log(zeta))
        T = np.linspace(T_c - Q/a - 1e-5, -10, 300)
        adT = a * (T-T_c)

        b_stripe = np.sqrt(-(adT + Q) / (3*b))
        ax.plot(phi_c + b_stripe, T, "--", color=color)
        ax.plot(phi_c - b_stripe, T, "--", color=color)

        ax.plot([], [], "--o", markerfacecolor="w", color=color, label=f"$Y$ = {Y[0]:.0f}\\  kPa")

    ax.set_xlim(0, 0.8)
    ax.set_ylim(-10, 85)
    ax.set_xlabel(r"$\phi_{0}$")
    ax.set_ylabel(r"$T_\mathrm{m}$ (${}^{\circ}\mathrm{C}$)", labelpad=8)
    ax.text(*labelpos, r"\textbf{(c)}", transform=ax.transAxes, fontsize=9, ha="center")
    ax.legend()

    plt.tight_layout()
    plt.savefig("isotropic_scaling.pdf",
                crop=True,
                optimize=True,
                transparent=True,
                facecolor="none",
                pad_inches=0)
    plt.show()
