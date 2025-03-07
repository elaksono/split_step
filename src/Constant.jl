"""
Constant Module: Defines and exports fundamental physical constants for Quantum Optics and Electromagnetism\n
Updated: 6 Mar 2025\n

Constants:\n
  - c0: Speed of light in vacuum (299792458 m/s)
  - Z0: Impedance of free space (~376.730313461 Ω)
  - hbar: Reduced Planck's constant (1.0545718e-34 J·s)
  - e: Elementary charge (1.602176634e-19 C)
  - me: Electron mass (9.10938356e-31 kg)
  - eps0: Vacuum permittivity (8.854187817e-12 F/m)
  - mu0: Vacuum permeability (1.25663706212e-6 N/A²)
  - kB: Boltzmann constant (1.380649e-23 J/K)
"""
module Constant

export c0, Z0, hbar, e, me, eps0, mu0, kB

c0   = 299792458             # Speed of light in vacuum (m/s)
Z0   = 376.730313461         # Impedance of free space (Ohms)
hbar = 1.0545718e-34         # Reduced Planck's constant (J·s)
e    = 1.602176634e-19       # Elementary charge (C)
me   = 9.10938356e-31        # Electron mass (kg)
eps0 = 8.854187817e-12       # Vacuum permittivity (F/m)
mu0  = 1.25663706212e-6      # Vacuum permeability (N/A²)
kB   = 1.380649e-23          # Boltzmann constant (J/K)

end