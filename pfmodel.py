# -*- coding: utf-8 -*-

import numpy as np

class PFVarStiff:

    def __init__(self,
                 T,
                 T_c=70,
                 phi_c=0.2,
                 phi_0=0.5,
                 psi_0=None,
                 a=0.025,
                 b=2,
                 kappa=0.013,
                 M_field=4000,
                 h_field=0.5,
                 h_num=5,
                 n=128,
                 L=5,
                 dt=1,
                 disorder=0.1):
        """Elastic microphase separation with varying stiffness.

        Parameters
        ----------
        T : float
            Temperature (in Celsius).
        T_c : float
            Critical temperature (in Celsius).
        phi_c : float
            Critical polymer volume fraction.
        phi_0 : float
            Mean polymer volume fraction.
        psi_0 : 2d array of size (n, n)
            Initial psi.  If it's not given, a random initial condition
            with average equal to phi_0 is used.
        a : float
            Landau parameter a (in kPa).
        b : float
            Landau parameter b (in kPa).
        kappa : float
            Interfacial parameter (in kPa μm^2).
        n : int
            Number of grid points along each axis of the box.
        M_field : float or 2d array of size (n, n)
            Rescaled longitudinal modulus (in kPa)
        h_field : float or 2d array of size (n, n)
            Coarse-graining length (in μm).
        h_num : int
            Number of constant h fields to be used in approximating
            variable h coarse-graining.
        L : float
            side length of the box (in μm)
        dt : float
            time step
        disorder: float
            Controls the randomness in the initial configuration.
        """
        # Choose a random initial condition if none is given.  Too much
        # randomness can result in overflows, whereas too little
        # randomness will sometimes prevent the system from reaching its
        # true energy minimum.
        if psi_0 is None:
            self.psi = np.random.normal(size=(n, ) * 2, scale=disorder)
        else:
            self.psi = psi_0

        # Make sure that the mean value of the order parameter is exact.
        self.psi += phi_0 - phi_c - self.psi.mean()

        # Domain setup.
        x = np.linspace(-L, L, n)
        dx = x[1] - x[0]

        # Wavenumber arrays.  The wavenumbers need to be multiplied by
        # 2pi to get usual physics conventions.
        q = 2 * np.pi * np.fft.fftfreq(n, d=dx)
        q2 = q[:, None]**2 + q[None, :]**2

        # Coarse-graining kernel.
        X, Y = np.meshgrid(x, x)
        self.K_q_list = []
        h_list = np.linspace(h_field.min(), h_field.max(), h_num)
        for h in h_list:
            # Normalized kernel in real space.
            K = np.exp(-(X**2 + Y**2) / (4 * h**2))
            K /= (4 * np.pi * h**2)

            # DFTs assume that the "origin" of the kernel are at the "ends".
            # But the kernel we've defined above has an origin at the center.
            # So shift appropriately to put the origin at the "ends."
            K = np.fft.fftshift(K)
            # The multiplication by dx^2 is to turn a discrete DFT sum into
            # an integral.
            K_q = np.fft.fftn(K) * (dx**2)

            self.K_q_list.append(K_q)

        # Compute interpolation coefficient fields.
        if h_num == 1:
            self.coeff = [1.0]
        else:
            self.coeff = [np.zeros((n, n)) for _ in range(h_num)]

            # For each point in h_field, find the closest point in h_list
            # and compute the associated entries in the coefficient fields.
            for i in range(n):
                for j in range(n):
                    k = np.searchsorted(h_list, h_field[i][j])
                    if k <= 0:
                        self.coeff[0][i][j] = 1.0
                    elif k >= h_num:
                        self.coeff[-1][i][j] = 1.0
                    else:
                        self.coeff[k - 1][i][j] = (h_list[k] - h_field[i][j]) / (h_list[k] -
                                                                                 h_list[k - 1])
                        self.coeff[k][i][j] = (h_field[i][j] - h_list[k - 1]) / (h_list[k] -
                                                                                 h_list[k - 1])
        for i in range(h_num):
            self.coeff[i] *= M_field

        self.b = b
        self.h_num = h_num

        # Semi-implicit Euler for just the interface term.
        self.A = 1 - a * (T-T_c) * q2 * dt
        self.B = q2 * dt
        self.C = 1 + kappa*q2*q2*dt

    def evolve(self):
        """Evolves the energy-minimization equation in time."""
        psi_q = np.fft.fftn(self.psi)
        eta = self.b * self.psi * self.psi * self.psi

        for i in range(self.h_num):
            psi_h = self.coeff[i] * np.fft.ifftn(psi_q * self.K_q_list[i])

            psi_coeff_q = np.fft.fftn(self.psi * self.coeff[i])
            psi_h += np.fft.ifftn(psi_coeff_q * self.K_q_list[i])

            eta += 0.5 * psi_h.real

        eta_q = np.fft.fftn(eta)
        psi_q = (self.A * psi_q - self.B * eta_q) / self.C
        self.psi = np.fft.ifftn(psi_q).real

class PFUniaxial:

    def __init__(self,
                 T,
                 T_c=70,
                 phi_c=0.2,
                 phi_0=0.5,
                 psi_0=None,
                 a=0.025,
                 b=2,
                 kappa=0.013,
                 M=4000,
                 h=0.5,
                 p=1,
                 d=2,
                 n=128,
                 L=5,
                 dt=1,
                 disorder=0.1):
        """Phase-field model for elastic microphase separation.

        Parameters
        ----------
        T : float
            Temperature (in Celsius).
        T_c : float
            Critical temperature (in Celsius).
        phi_c : float
            Critical polymer volume fraction.
        psi_0 : float
            Initial psi.
        a : float
            Landau parameter a (in kPa).
        b : float
            Landau parameter b (in kPa).
        kappa : float
            Interfacial parameter (in kPa μm^2).
        M : float
            Rescaled longitudinal modulus (in kPa)
        h : float
            Coarse-graining length (in μm).
        d : int
            Dimension of box used for energy minimization.
        n : int
            Number of grid points along each axis of the box.
        L : float
            side length of the box (in μm)
        dt : float
            time step
        disorder: float
            Controls the randomness in the initial configuration.
        """
        if d not in (2, 3):
            raise ValueError(f"The dimension (d = {d}) must be 2, or 3.")

        # Choose a random initial condition if none is given.  Too much
        # randomness can result in overflows, whereas too little
        # randomness will sometimes prevent the system from reaching its
        # true energy minimum.
        if psi_0 is None:
            self.psi = np.random.normal(size=(n, ) * d, scale=disorder)
        else:
            self.psi = psi_0

        # Make sure that the mean value of the order parameter is exact.
        self.psi += phi_0 - phi_c - self.psi.mean()

        # Domain setup.
        x = np.linspace(-L, L, n)
        dx = x[1] - x[0]

        # Wavenumber arrays.  The wavenumbers need to be multiplied by
        # 2pi to get usual physics conventions.
        q = 2 * np.pi * np.fft.fftfreq(n, d=dx)
        if d == 1:
            q2 = q**2
        elif d == 2:
            q2 = q[:, None]**2 + q[None, :]**2
        else:
            q2 = q[:, None, None]**2 + q[None, :, None]**2 + q[None, None, :]**2

        # Coarge-graining lengths.
        h_x = h * p
        h_y = h / np.sqrt(p)
        h_z = h / np.sqrt(p)

        # Coarse-graining kernel.
        if d == 2:
            X, Y = np.meshgrid(x, x)
            K = np.exp(-(X * X / (4 * h_x**2) + Y * Y / (4 * h_y**2)))
        else:
            X, Y, Z = np.meshgrid(x, x, x)
            K = np.exp(-(X * X / (4 * h_x**2) + Y * Y / (4 * h_y**2) + Z * Z / (4 * h_z**2)))
            # Normalization.
            K /= np.sqrt(4 * np.pi * h_z**2)

        # Common normalization.
        K /= np.sqrt(4 * np.pi * h_x**2)
        K /= np.sqrt(4 * np.pi * h_y**2)

        # DFTs assume that the "origin" of the kernel are at the "ends".
        # But the kernel we've defined above has an origin at the center.
        # So shift appropriately to put the origin at the "ends."
        K = np.fft.fftshift(K)
        # The multiplication by dx^d is to turn a discrete DFT sum into
        # an integral.
        K_q = np.fft.fftn(K) * (dx**d)

        M_q = M * ((p**2 - 1/p) * (q**2 + 1e-10) / (q2+1e-10) + 1/p)
        #M_q = M/p

        # Precomputable stuff that's used in each step.
        self.A = 1 - 3 * a * (T-T_c) * q2 * dt
        self.B = b * q2 * dt
        self.C = 1 + dt * q2 * (kappa*q2 - 2 * a * (T-T_c) + M_q*K_q)

    def evolve(self):
        """Evolves the energy-minimization equation in time."""
        psi_q = np.fft.fftn(self.psi)
        psi3_q = np.fft.fftn(self.psi**3)

        psi_q = (self.A * psi_q - self.B * psi3_q) / self.C
        self.psi = np.fft.ifftn(psi_q).real
