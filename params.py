"""Physical parameters used in the model."""

import numpy as np

# Fundamental constants.
kB = 1.380649e-23 # Boltzmann constant (J/K)
lSiO = 1.64e-10 # Si-O bond length (m)
ell = 2 * lSiO # repeat unit length (m)
NA = 6.02214076e23 # Avogadro's number

# Physical parameters.
T = 300 # Temperature (Kelvin)
kBT = kB * T # Molecular energy scale (J)

# PDMS parameters.
# Dry PDMS density (kg/m^3):
# https://www.gelest.com/product/DMS-V31
# (The above link gives the relative density as 0.97.)
rho = 970
# Repeat unit (CH3)2SiO molar mass:
# https://www.webqc.org/molecular-weight-of-(CH3)2SiO.html
m_r = 0.0742 # in kg/mol
m_r /= NA # convert to kg.
C_inf = 6.8 # characteristic ratio of PDMS (assumed) (dimensionless)

# Calculated dimensionful quantities.
kappa = kBT / ell # Interfacial parameter (J/m)

# Model parameters.
# scale factor (for the mesh size)
n = 35

# Estimated parameters.
B = rho * ell**2 * C_inf * kBT / m_r

# Guessed parameters.
phi_c = 0.2 # critical volume fraction

# The rescaled longitudinal modulus M (in J/m^3) as a function of the Young's
# modulus Y of the dry PDMS in kPa (= 1000 J/m^3) and the swollen
# network volume fraction phi_c.
def longitudinal_modulus(Y):
    G = Y * 1000 / 3 # Young's modulus to shear modulus in J/m^3
    return phi_c**(-2) * (phi_c + phi_c**(1/3)) * G

# Mesh size (in m) as a function of the Young's modulus Y of the
# dry PDMS in kPa (= 1000 J/m^3) and the swollen network
# volume fraction phi_c.
def mesh_size(Y):
    Y = Y * 1000
    return np.sqrt(3 * B / Y) * phi_c**(-1 / 3)

# Return cross-link density times kbT as a function of the
# Young's modulus in kPa
def nukBT(Y):
    M = longitudinal_modulus(Y)
    return M * phi_c ** 2 / (phi_c + phi_c ** (1/3))

# Coarse-graining length scale (in m) as a function of the Young's
# modulus Y of the dry PDMS in kPa (= 1000 J/m^3) and the swollen
# network volume fraction phi_c.
def cg_length(Y):
    return n * mesh_size(Y)

# The parameter ζ as a function of the Young's modulus Y of the
# dry PDMS in kPa (= 1000 J/m^3) and the swollen network
# volume fraction phi_c.
def zeta(Y):
    M = longitudinal_modulus(Y)
    h = cg_length(Y)

    return M * h**2 / kappa

# Domain size Λ (in m) as a function of the Young's modulus of the dry
# PDMS in kPa (= 1000 J/m^3) and the swollen network volume fraction
# phi_c.
def domain_size(Y):
    h = cg_length(Y)
    z = zeta(Y)

    q_m = np.sqrt(np.log(z)) / h
    return 2 * np.pi / q_m

def num_monomers(Y):
    G = Y * 1000 / 3 # Young's modulus to shear modulus in J/m^3
    return rho * kBT / (G*m_r)

def domain_size_eq(Y):
    Y = Y * 1000
    l = (2*np.pi) ** 2
    l *= 3*B*n**2
    l /= Y
    l /= np.log(B*n**2/kappa*(phi_c**(-7/3) + phi_c**(-5/3)))
    l *= phi_c**(-2/3)

    l = np.sqrt(l)
    return l

def print_params(Y):
    print("---------------------")
    print(f"Parameters for Y = {Y} kPa:")
    print(f"  Critical phi_c = {phi_c:.3}")
    print(f"  Energy scale = {kBT:.3} J")
    print(f"  Parameter B = {B:.3} Pa m^2")
    print(f"  Paramer B (approx) = {B:.2} Pa m^2")
    print(f"  PDMS repeat unit length = {ell:.3} m")
    print(f"  Mass of PDMS repeat unit = {m_r:.3} kg")
    print(f"  Flory ratio for PDMS = {C_inf:.3}")
    print(f"  Longitudinal modulus/M = {longitudinal_modulus(Y):.3}")
    print(f"  nukBT = {nukBT(Y):.3}")
    print(f"  Elastocapillary = {B/kappa*(phi_c**(-7/3) + phi_c**(-5/3))}")
    print(f"  Num repeat units = {num_monomers(Y):.3}")
    print(f"  Mesh size = {mesh_size(Y)/1e-9:.3} nm")
    print(f"  Domain size = {domain_size(Y)/1e-6:.3} um")
    print(f"  Domain size (eq) = {domain_size_eq(Y)/1e-6:.3} um")
    print("---------------------")

def parameters(Y, dim=False, length_scale=1e-6, energy_scale=1e-15, kwargs=False):
    if dim:
        epl3 = epl = l = 1
    else:
        # Nondimensionalization.
        epl3 = energy_scale / length_scale**3 # in J/m^3
        epl = energy_scale / length_scale # in J/m
        l = length_scale

    params = {
        "kappa": kappa / epl,
        "h": n * mesh_size(Y) / l,
        "L": 7 * domain_size(Y) / l,
        "nukbt": nukBT(Y) / epl3
    }

    if kwargs:
        return params
    else:
        return params["kappa"], params["h"], params["nukbt"], zeta(Y)

if __name__ == "__main__":
    #print_params(800)
    print_params(350)
    #print_params(10)
