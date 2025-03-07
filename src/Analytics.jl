"""
Analytics Module: Analytical OPA/OPO Solutions and Bandwidth Calculations\n
Updated: 6 Mar 2025\n

This module provides a suite of functions to analyze optical parametric amplification (OPA) 
and optical parametric oscillation (OPO) processes using both analytical and numerical methods.
It is tailored for systems with quasi-phase matching (QPM) and includes tools for handling 
both periodically poled and linearly chirped waveguides.\n
\n
Key functionalities include:\n

1. Analytical OPA Intensity:
   - opa_cw_intensity(z, dk, opa):
       Computes the intensities (signal, idler, and pump) along the propagation distance 'z'
       using an analytical solution based on Jacobi elliptic functions (specifically, the "cn" function).

2. OPA ODE Solver:
   - opa_cw_ode(zvec, dk, opa):
       Defines and solves the coupled ordinary differential equations (ODEs) governing the 
       evolution of the complex amplitudes in an OPA process. It uses DifferentialEquations.jl 
       for numerical integration and returns the solution in terms of amplitudes and flux.

3. CW OPO Amplitude Threshold:
   - cth_cw_opo(loss, kappa, L):
       Calculates the amplitude threshold required for continuous-wave OPO operation in 
       periodically poled devices based on material nonlinearity and waveguide length.

4. Linearly Chirped OPA Functions:
   - dk_linear_opa(z, dω, omg_ctr, qpm_lambda, mat, phi_z_pars):
       Computes the phase mismatch for a linearly chirped OPA system. The chirp is determined 
       by a linear variation in the poling period, characterized by the parameters (zc, dk_1).

   - solve_bw_con(omg_ctr, qpm_lambda, mat; phi_z_pars, dω0):
       Solves for the gain bandwidth in frequency for a periodically poled waveguide. The bandwidth 
       is determined where the sinc function (arising from phase mismatch) reaches zero.

   - solve_bw_lbd_con(omg_ctr, qpm_lambda, mat; phi_z_pars, dω0, ids):
       Converts the frequency bandwidth solution into wavelength terms and returns the bandwidth 
       along with the corresponding edge wavelengths and frequencies.

   - solve_dω_for_z(z, omg_ctr, qpm_lambda, mat, phi_z_pars; dω_guess):
       Finds the frequency offset (dω) for which the phase matching condition is met at a given 
       propagation distance 'z' in a linearly chirped OPA setup.

   - solve_bw_linear(omg_ctr, qpm_lambda, mat, phi_z_pars):
       Computes the overall gain bandwidth (in frequency) of a linearly chirped OPA by evaluating 
       the frequency shift between the beginning and end of the waveguide.

   - solve_bw_lbd_linear(omg_ctr, qpm_lambda, mat, phi_z_pars; ids):
       Similar to solve_bw_linear, but the result is presented in terms of wavelength bandwidth 
       along with additional spectral details.

   - cth_cw_opo_lin(loss, kappa, dk_1):
       Determines the amplitude threshold for CW OPO operation in linearly poled systems, taking 
       into account the chirp rate of the phase mismatch.

Dependencies:
   - Constant: Provides physical constants.
   - Elliptic and Elliptic.Jacobi: Offer computations for Jacobi elliptic functions used in the 
     analytical OPA solution.
   - DifferentialEquations: Supplies numerical solvers for the OPA ODEs.
   - NLOclass: Contains definitions and functions for nonlinear optical processes.
   - Materials: Supplies material properties and dispersion functions.
   - Roots: Used for root-finding in phase-matching and bandwidth calculations.

Overall, this module is designed to facilitate the analysis of nonlinear optical interactions 
in engineered waveguide structures, providing both direct analytical expressions and numerical 
tools to simulate and optimize OPA and OPO processes under various experimental conditions.
"""
module Analytics

using Constant
using Elliptic
using Elliptic.Jacobi
using DifferentialEquations
using NLOclass
using Materials
using Roots

export opa_cw_intensity, opa_cw_ode, cth_cw_opo
"""
Analytical solution for OPA with QPM
- Jacobi Elliptic function "cn"
"""
function opa_cw_intensity(z, dk, opa::nl_class)
    (Ns0, Ni0, Np0) = abs.(opa.init_amp[1,:]).^2
    αt = (Ns0 + Np0) +  1/4*(dk/opa.mat.kappa)^2
    βt = sqrt(αt^2 - (dk/opa.mat.kappa)^2 * Np0)
    Np0t = Np0 - (αt - βt)/2
    γt = Np0t / βt
    
    uz = sqrt(βt) * opa.mat.kappa * z + Elliptic.K(γt)
    
    Ns = Ns0 + Np0t * Jacobi.cn(uz, γt )^2
    Ni = Ni0 + Np0t * Jacobi.cn(uz, γt )^2
    Np = Np0 - Np0t * Jacobi.cn(uz, γt )^2
    return [Ns, Ni, Np]
end;

"""
The ODE for OPA
"""
function opa_cw_ode(zvec, dk, opa::nl_class)
    function f_opa!(du, u, p, z)
        du[1] =  p[1] * u[3] * conj(u[2]) * opa.nl_func(opa.phi_z(z))
        du[2] =  p[1] * u[3] * conj(u[1]) * opa.nl_func(opa.phi_z(z))
        du[3] = -p[1] * u[1] * u[2] * conj(opa.nl_func(opa.phi_z(z)))
    end
    
    zspan = (minimum(zvec), maximum(zvec))
    pars = [opa.mat.kappa]
    prob = ODEProblem(f_opa!, opa.init_amp[1,:], zspan, pars, saveat = zvec, dtmax = 1e-2)
    sol = solve(prob)
    amp = hcat(sol.u...)
    flux = abs.(amp) .^ 2
    return Dict("t" => sol.t, "amp" => amp, "flux" => flux)
end;

"""
Amplitude threshold for CW OPO (periodically poled)
"""
function cth_cw_opo(loss, kappa, L)
    return atanh(sqrt(loss)) / (kappa * L)
end

# Linearly chirped OPA
export dk_linear_opa, solve_dω_for_z, solve_bw_con, solve_bw_lbd_con, solve_bw_linear, solve_bw_lbd_linear, cth_cw_opo_lin
"""
dk_linear_opa(z::Real, dω::Real, omg_ctr::Vector, qpm_lambda::Real, mat::Material, 
        phi_z_pars::Tuple)
The phase mismatch in linearly chirped OPA

phi_z_pars: (zc, dk_1) --> (PM point for omg_ctr, poling chirp rate)
"""
function dk_linear_opa(z::Real, dω::Real, omg_ctr::Vector, qpm_lambda::Real, mat::Material, 
        phi_z_pars::Tuple)
    (zc, dk_1) = phi_z_pars
    delta_beta = (beta_func(mat, omg_ctr[3]) - beta_func(mat, omg_ctr[1] + dω) 
        - beta_func(mat, omg_ctr[2] - dω))
    dkz = dk_1 * (z-zc) + (delta_beta - 2*pi/(qpm_lambda * 1e-3))
    return dkz
end;

"""
solve_bw_con(omg_ctr::Vector, qpm_lambda::Real, mat::Material; phi_z_pars = (0.0,0.0), dω0 = 10.0)

Solving the bandwidth for a periodically poled waveguide
--> This corresponds to the width at which sinc(0.5*dk*L) = 0
"""
function solve_bw_con(omg_ctr::Vector, qpm_lambda::Real, mat::Material; phi_z_pars = (0.0,0.0), dω0 = 0.0)
    fp(dω; m = 1) = 0.5 * dk_linear_opa(0, dω, omg_ctr, qpm_lambda, mat, phi_z_pars) * mat.prop.ll - m * pi
    fm(dω; m = 1) = 0.5 * dk_linear_opa(0, dω, omg_ctr, qpm_lambda, mat, phi_z_pars) * mat.prop.ll + m * pi
    omega_1 = find_zero(fp, dω0)
    omega_2 = find_zero(fm, dω0)
    return abs(omega_1 - omega_2)
end;

function solve_bw_lbd_con(omg_ctr::Vector, qpm_lambda::Real, mat::Material; phi_z_pars = (0.0,0.0), dω0 = 0.0, ids = 1)
    fp(dω; m = 1) = 0.5 * dk_linear_opa(0, dω, omg_ctr, qpm_lambda, mat, phi_z_pars) * mat.prop.ll - m * pi
    fm(dω; m = 1) = 0.5 * dk_linear_opa(0, dω, omg_ctr, qpm_lambda, mat, phi_z_pars) * mat.prop.ll + m * pi
    omega_1 = omg_ctr[ids] .+ find_zero(fp, dω0)
    omega_2 = omg_ctr[ids] .+ find_zero(fm, dω0)
    lbd_1 = (2*pi*c0) / (omega_1 * 1e12) * 1e6
    lbd_2 = (2*pi*c0) / (omega_2 * 1e12) * 1e6
    return Dict("bw_lbd" => abs(lbd_2 - lbd_1), "lbd_1" => lbd_1, "lbd_2" => lbd_2, 
        "bw_freq" => abs(omega_2 - omega_1)/(2*pi), "freq_1" => omega_1/(2*pi), "omega_2" => omega_2/(2*pi))
end;

# Define a function to solve for the root with respect to z
"""
solve_dω_for_z(z::Real, omg_ctr::Vector, qpm_lambda::Real, mat::Material, phi_z_pars::Tuple; dω_guess::Real = 0.0)

Solving the corresponding frequency (ωs + dω) for the PM at different z for linearly chirped OPA
"""
function solve_dω_for_z(z::Real, omg_ctr::Vector, qpm_lambda::Real, mat::Material, phi_z_pars::Tuple; dω_guess::Real = 0.0)
    f(dω) = dk_linear_opa(z, dω, omg_ctr, qpm_lambda, mat, phi_z_pars)
    dω_root = find_zero(f, dω_guess)  # Provide a guess for dω
    return dω_root
end;

"""
solve_bw_linear(omg_ctr::Vector, qpm_lambda::Real, mat::Material, phi_z_pars::Tuple)
Solve the gain bandwidth of the linearly chirped OPA
"""
function solve_bw_linear(omg_ctr::Vector, qpm_lambda::Real, mat::Material, phi_z_pars::Tuple)
    omega_1 = solve_dω_for_z(0.0, omg_ctr, qpm_lambda, mat, phi_z_pars)
    omega_2 = solve_dω_for_z(mat.prop.ll, omg_ctr, qpm_lambda, mat, phi_z_pars)
    return abs(omega_2 - omega_1)
end;

function solve_bw_lbd_linear(omg_ctr::Vector, qpm_lambda::Real, mat::Material, phi_z_pars::Tuple; ids = 1)
    omega_1 = solve_dω_for_z(0.0, omg_ctr, qpm_lambda, mat, phi_z_pars)
    omega_2 = solve_dω_for_z(mat.prop.ll, omg_ctr, qpm_lambda, mat, phi_z_pars)
    
    # The actual omega
    omega_1 = omg_ctr[ids] + omega_1
    omega_2 = omg_ctr[ids] + omega_2
    lbd_1 = (2*pi*c0) / (omega_1 * 1e12) * 1e6
    lbd_2 = (2*pi*c0) / (omega_2 * 1e12) * 1e6
    return Dict("bw_lbd" => abs(lbd_2 - lbd_1), "lbd_1" => lbd_1, "lbd_2" => lbd_2, 
        "bw_freq" => abs(omega_2 - omega_1)/(2*pi), "freq_1" => omega_1/(2*pi), "omega_2" => omega_2/(2*pi))
end;

"""
Amplitude threshold for CW OPO (linearly poled)
"""
function cth_cw_opo_lin(loss, kappa, dk_1)
    return (1/kappa) * sqrt(dk_1/(2*pi) * log(1/(1-loss)))
end

end