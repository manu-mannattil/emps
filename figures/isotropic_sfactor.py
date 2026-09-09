# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import charu
from params import parameters

data = np.loadtxt("../experiments/sfactor.dat")
temp = np.array([59.907, 58.237, 56.567, 54.897, 53.227, 51.557, 49.887, 48.217, 46.547])

q = 1e3*data[0] # wavenumber um^-1

rc = {
    "charu.doc": "aps",
    "charu.tex": True,
    "charu.tex.font": "fourier",
    "axes.axisbelow": False,
}

with plt.rc_context(rc):
    fig, ax = plt.subplots()

    # Parameters in the Landau energy (estimated).
    T_c = 70
    phi_c = 0.2
    a = 2.5e-2 # kPa/K
    b = 2 # kP3*a*B*n**2a
    B = 0.024
    n = 35

    # Other parameters.
    Y = 350 # Young's modulus in kPa
    kappa, h, nukbt, zeta = parameters(Y)
    M = nukbt*phi_c**(-2)*(phi_c + phi_c**(1/3))

    # Mean order parameter and temperature.
    psi_0 = 0.47110599 - phi_c # phi_0 at T_swell = 60 and Y = 350 kPa.

    # Normalize the experimental I(q) curves such that the peak at the
    # lowest temperature is unity.
    S_norm = data[9][20:].max()
    i = data[9][20:].argmax()
    for i in range(1, 9 + 1):
        ax.plot(q, data[i]/S_norm, "C0--", alpha=0.75)
    ax.plot([], [], "C0--", alpha=0.75, label="experiment")

    # Normalize the theoretical S(q) curves such that the peak at the
    # lowest temperature is unity.
    q_max = np.sqrt(np.log(zeta))/h
    print(f"q_max (theory) = {q_max}")
    print(f"T_m (theory) = {T_c - 3*b/a*psi_0**2 - Y*((kappa*phi_c**(2/3))/(3*a*B*n**2))*(1 + np.log(zeta))}")
    S_norm = a*(temp[-1] - T_c) + 3*b*psi_0**2 + kappa*q_max**2 + M * np.exp(-h**2*q_max**2)

    q = np.linspace(0, 16, 1000)
    for T in temp:
        tau = 2/kappa*(a*(T - T_c) + 3*b*psi_0**2 + M)
        S_q = a*(T - T_c) + 3*b*psi_0**2 + kappa*q**2 + M * np.exp(-h**2*q**2)
        S_q /= S_norm
        S_q = 1/S_q

        ax.plot(q, S_q, color="C3")

    ax.plot([], [], "C3-", label="theory")
    ax.set_ylabel(r"$S(q)$")
    ax.set_xlabel(r"$q$ ($\textsf{\textmu m}^{-1})$")

    ax.arrow(
        x=0.75, y=0.25, dx=0, dy=0.40, transform=ax.transAxes,
        head_width=0.015, head_length=0.06, length_includes_head=True,
        lw=0.25,
        color="#666666",
    )

    ax.text(0.72, 0.42,
            r"cooling",
            transform=ax.transAxes,
            va="center",
            ha="center",
            rotation=90,
            color="#666666")

    ax.plot([q_max, q_max], [-10, 10], "k--")
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, 16)
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        "isotropic_sfactor.pdf",
        crop=True,
        optimize=True,
        transparent=True,
        facecolor="none",
        pad_inches=0,
    )
    plt.show()
