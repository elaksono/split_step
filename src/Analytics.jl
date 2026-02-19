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

# Simple Numerical Differentiation
export D1f, D2f, D3f, D4f
function D1f(z, f::Function; dz = 1e-3)
    return (f(z + dz/2) - f(z - dz/2))/dz
end;

function D2f(z, f::Function; dz = 1e-3)
    return (f(z + dz) + f(z - dz) - 2 * f(z))/(dz^2)
end;

function D3f(z, f::Function; dz = 1e-3)
    out = f(z + 2*dz) - 2 * f(z + dz) + 2 * f(z - dz) - f(z - 2*dz)
    return out / (2*dz^3)
end;

function D4f(z, f::Function; dz = 1e-3)
    out = f(z + 2*dz) - 4 * f(z + dz) + 6 * f(z) - 4 * f(z - dz) + f(z - 2*dz)
    return out / (dz^4)
end;

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
export dk_linear_opa, solve_ωs_for_z, solve_bw_con, solve_bw_con_full, solve_bw_linear, solve_bw_linear_full, cth_cw_opo_lin
"""
dk_linear_opa(z::Real, dω::Real, omg_ctr::Vector, qpm_lambda::Real, mat::Material, phi_z_pars::Tuple)
dk_linear_opa(z::Real, dω::Real, pulse_adb::nl_pulse_adb)
    The phase mismatch in linearly chirped OPA

phi_z_pars: (zc, dk_1) --> (PM point for omg_ctr, poling chirp rate)
"""
function dk_linear_opa(z::Real, dω::Real, omg_ctr::Vector, qpm_lambda::Real, mat::Material, phi_z_pars::Tuple)
    (zc, dk_1) = phi_z_pars
    if typeof(mat) <: TaylorMaterial
        delta_beta = (beta_func(mat, omg_ctr[3], 3) - beta_func(mat, omg_ctr[1] + dω, 1) - beta_func(mat, omg_ctr[2] - dω, 2))
    else
        delta_beta = (beta_func(mat, omg_ctr[3]) - beta_func(mat, omg_ctr[1] + dω) - beta_func(mat, omg_ctr[2] - dω))
    end
        
    dkz = dk_1 * (z-zc) + (delta_beta - 2*pi/(qpm_lambda * 1e-3))
    return dkz
end;

function dk_linear_opa(z::Real, dω::Real, pulse_adb::nl_pulse_adb)
    (zc, dk_1) = pulse_adb.phi_z_pars
    mat = pulse_adb.mat
    omg_ctr = pulse_adb.pulse.omg_ctr
    if typeof(mat) <: TaylorMaterial
        delta_beta = (beta_func(mat, omg_ctr[3], 3) - beta_func(mat, omg_ctr[1] + dω, 1) - beta_func(mat, omg_ctr[2] - dω, 2))
    else
        delta_beta = (beta_func(mat, omg_ctr[3]) - beta_func(mat, omg_ctr[1] + dω) - beta_func(mat, omg_ctr[2] - dω))
    end
    
    dkz = dk_1 * (z-zc) + (delta_beta - pulse_adb.dk_qpm)
    return dkz
end;

"""
solve_bw_con(omg_ctr::Vector, qpm_lambda::Real, mat::Material; dω0 = 0.0, zqpm = 0.0)
solve_bw_con(pulse_adb::nl_pulse_adb; dω0 = 0.0, zqpm = 0.0)
    Return: Δω

Solving the bandwidth for a periodically poled waveguide (CONST QPM)
--> This corresponds to the width at which sinc(0.5*dk*L) = 0
--> phi_z_pars is set to (0.0, 0.0) and dk is sampled at z = 0
"""
function solve_bw_con(omg_ctr::Vector, qpm_lambda::Real, mat::Material; dω0 = 0.0, zqpm = 0.0)
    fp(dω; m = 1) = 0.5 * dk_linear_opa(zqpm, dω, omg_ctr, qpm_lambda, mat, (0.0,0.0)) * mat.prop.ll - m * pi
    fm(dω; m = 1) = 0.5 * dk_linear_opa(zqpm, dω, omg_ctr, qpm_lambda, mat, (0.0,0.0)) * mat.prop.ll + m * pi
    omega_1 = find_zero(fp, dω0)
    omega_2 = find_zero(fm, dω0)
    return abs(omega_1 - omega_2)
end;

function solve_bw_con(pulse_adb::nl_pulse_adb; dω0 = 0.0, zqpm = 0.0)
    omg_ctr = pulse_adb.pulse.omg_ctr
    qpm_lambda = (2*pi) / (pulse_adb.dk_qpm) * 1e3
    mat = pulse_adb.mat
    fp(dω; m = 1) = 0.5 * dk_linear_opa(zqpm, dω, omg_ctr, qpm_lambda, mat, (0.0,0.0)) * mat.prop.ll - m * pi
    fm(dω; m = 1) = 0.5 * dk_linear_opa(zqpm, dω, omg_ctr, qpm_lambda, mat, (0.0,0.0)) * mat.prop.ll + m * pi
    omega_1 = find_zero(fp, dω0)
    omega_2 = find_zero(fm, dω0)
    return abs(omega_1 - omega_2)
end;

"""
solve_bw_con_full(omg_ctr::Vector, qpm_lambda::Real, mat::Material; dω0 = 0.0, zqpm = 0.0, ids = 1)
solve_bw_con_full(pulse_adb::nl_pulse_adb; dω0 = 0.0, zqpm = 0.0, ids = 1)
    ids - The index for the signal
    Return: Dict ("bwd_lbd", "lbd_1", "lbd_2", "bw_omega", "omega_1", "omega_2", "bw_freq", "freq_1", freq_2")

Solving the bandwidth for a periodically poled waveguide (CONST QPM)
--> This corresponds to the width at which sinc(0.5*dk*L) = 0
--> phi_z_pars is set to (0.0, 0.0) and dk is sampled at z = 0
"""
function solve_bw_con_full(omg_ctr::Vector, qpm_lambda::Real, mat::Material; dω0 = 0.0, zqpm = 0.0, ids = 1)
    fp(dω; m = 1) = 0.5 * dk_linear_opa(zqpm, dω, omg_ctr, qpm_lambda, mat, (0.0,0.0)) * mat.prop.ll - m * pi
    fm(dω; m = 1) = 0.5 * dk_linear_opa(zqpm, dω, omg_ctr, qpm_lambda, mat, (0.0,0.0)) * mat.prop.ll + m * pi
    omega_1 = omg_ctr[ids] .+ find_zero(fp, dω0)
    omega_2 = omg_ctr[ids] .+ find_zero(fm, dω0)
    lbd_1 = (2*pi*c0) / (omega_1 * 1e12) * 1e6
    lbd_2 = (2*pi*c0) / (omega_2 * 1e12) * 1e6
    return Dict("bw_lbd" => abs(lbd_2 - lbd_1), "lbd_1" => lbd_1, "lbd_2" => lbd_2, 
        "bw_omega" => abs(omega_2 - omega_1), "omega_1" => omega_1, "omega_2" => omega_2,
        "bw_freq" => abs(omega_2 - omega_1)/(2*pi), "freq_1" => omega_1/(2*pi), "freq_2" => omega_2/(2*pi))
end;

function solve_bw_con_full(pulse_adb::nl_pulse_adb; dω0 = 0.0, zqpm = 0.0, ids = 1)
    ll = pulse_adb.mat.prop.ll
    fp(dω; m = 1) = 0.5 * dk_linear_opa(zqpm, dω, pulse_adb) * ll - m * pi
    fm(dω; m = 1) = 0.5 * dk_linear_opa(zqpm, dω, pulse_adb) * ll + m * pi
    
    omega_1 = omg_ctr[ids] .+ find_zero(fp, dω0)
    omega_2 = omg_ctr[ids] .+ find_zero(fm, dω0)
    lbd_1 = (2*pi*c0) / (omega_1 * 1e12) * 1e6
    lbd_2 = (2*pi*c0) / (omega_2 * 1e12) * 1e6
    return Dict("bw_lbd" => abs(lbd_2 - lbd_1), "lbd_1" => lbd_1, "lbd_2" => lbd_2, 
        "bw_omega" => abs(omega_2 - omega_1), "omega_1" => omega_1, "omega_2" => omega_2,
        "bw_freq" => abs(omega_2 - omega_1)/(2*pi), "freq_1" => omega_1/(2*pi), "freq_2" => omega_2/(2*pi))
end;

"""
solve_dω_for_z(z::Real, omg_ctr::Vector, qpm_lambda::Real, mat::Material, phi_z_pars::Tuple; dω_guess::Real = 0.0)
solve_ωs_for_z(z::Real, pulse_adb::nl_pulse_adb; dω0 = 0.0, ids = 1)
    Solving the corresponding frequency ωs for the PM at different z for linearly chirped OPA
"""
function solve_ωs_for_z(z::Real, omg_ctr::Vector, qpm_lambda::Real, mat::Material, phi_z_pars::Tuple; dω0 = 0.0, ids = 1)
    f(dω) = dk_linear_opa(z, dω, omg_ctr, qpm_lambda, mat, phi_z_pars)
    ωs = omg_ctr[ids] .+ find_zero(f, dω0)  # Provide a guess for dω
    return ωs
end;

function solve_ωs_for_z(z::Real, pulse_adb::nl_pulse_adb; dω0 = 0.0, ids = 1)
    f(dω) = dk_linear_opa(z, dω, pulse_adb)
    ωs = pulse_adb.pulse.omg_ctr[ids] .+ find_zero(f, dω0)  # Provide a guess for dω
    return ωs
end;

"""
solve_bw_linear(omg_ctr::Vector, qpm_lambda::Real, mat::Material, phi_z_pars::Tuple; dω_init = (0.0, 0.0), ids = 1)
solve_bw_linear(pulse_adb::nl_pulse_adb; dω_init = (0.0, 0.0), ids = 1)
    Solve the gain bandwidth of the linearly chirped OPA
    Return: Δω
"""
function solve_bw_linear(omg_ctr::Vector, qpm_lambda::Real, mat::Material, phi_z_pars::Tuple; dω_init = (0.0, 0.0), ids = 1)
    omega_1 = solve_ωs_for_z(0.0, omg_ctr, qpm_lambda, mat, phi_z_pars; dω0 = dω_init[1], ids = ids)
    omega_2 = solve_ωs_for_z(mat.prop.ll, omg_ctr, qpm_lambda, mat, phi_z_pars; dω0 = dω_init[2], ids = ids)
    return abs(omega_2 - omega_1)
end;

function solve_bw_linear(pulse_adb::nl_pulse_adb; dω_init = (0.0, 0.0), ids = 1)
    omega_1 = solve_ωs_for_z(0.0, pulse_adb; dω0 = dω_init[1], ids = ids)
    omega_2 = solve_ωs_for_z(pulse_adb.mat.prop.ll, pulse_adb; dω0 = dω_init[2], ids = ids)
    return abs(omega_2 - omega_1)
end;

"""
solve_bw_linear(pulse_adb::nl_pulse_adb; dω_init = (0.0, 0.0), ids = 1)
    Solve the gain bandwidth of the linearly chirped OPA
    Return: Dict ("bwd_lbd", "lbd_1", "lbd_2", "bw_omega", "omega_1", "omega_2", "bw_freq", "freq_1", freq_2")
"""
function solve_bw_linear_full(pulse_adb::nl_pulse_adb; dω_init = (0.0, 0.0), ids = 1, idp = 3, idt = 1)
    dk_1 = pulse_adb.phi_z_pars[2]
    gamma = pulse_adb.mat.kappa * abs(pulse_adb.init_amp[idt, idp])
    Lg = 4*gamma / abs(dk_1)
    
    if Lg > pulse_adb.mat.prop.ll
        throw(DomainError(Lg, "Approximation invalid: Amplification length is longer than the crystal length (Lg > L)"))
    end
    
    omega_1 = solve_ωs_for_z(Lg/2, pulse_adb; dω0 = dω_init[1], ids = ids)
    omega_2 = solve_ωs_for_z(pulse_adb.mat.prop.ll - Lg/2, pulse_adb; dω0 = dω_init[2], ids = ids)    
    lbd_1 = (2*pi*c0) / (omega_1 * 1e12) * 1e6
    lbd_2 = (2*pi*c0) / (omega_2 * 1e12) * 1e6
    return Dict("bw_lbd" => abs(lbd_2 - lbd_1), "lbd_1" => lbd_1, "lbd_2" => lbd_2, 
        "bw_omega" => abs(omega_2 - omega_1), "omega_1" => omega_1, "omega_2" => omega_2,
        "bw_freq" => abs(omega_2 - omega_1)/(2*pi), "freq_1" => omega_1/(2*pi), "freq_2" => omega_2/(2*pi));
end;

export gain_rsb_omg, gain_rsb_lbd, phase_rsb_lbd
"""
gain_rsb_omg(omega, pulse_adb::nl_pulse_adb; idt = 1, idp = 3)
    idt, idp - picking on the pump amplitude
    Rosenbluth gain Exp[pi*gamma^2/dk']
"""
function gain_rsb_omg(omega, pulse_adb::nl_pulse_adb; idt = 1, idp = 3)
    bw_dic = solve_bw_linear_full(pulse_adb)
    (omega_1, omega_2) = minmax(bw_dic["omega_1"], bw_dic["omega_2"])
    dk_1 = pulse_adb.phi_z_pars[2]
    gamma = pulse_adb.mat.kappa * abs(pulse_adb.init_amp[idt, idp])
    gain = 1.0
    if (omega_1 <= omega <= omega_2)
        gain = exp(pi*gamma^2 / dk_1)
    end
    
    return gain
end;

"""
gain_rsb_lbd(lbd, pulse_adb::nl_pulse_adb; idt = 1, idp = 3)
    idt, idp - picking on the pump amplitude
    Rosenbluth gain Exp[pi*gamma^2/dk']
"""
function gain_rsb_lbd(lbd, pulse_adb::nl_pulse_adb; idt = 1, idp = 3)
    bw_dic = solve_bw_linear_full(pulse_adb)
    (lbd_1, lbd_2) = minmax(bw_dic["lbd_1"], bw_dic["lbd_2"])
    dk_1 = pulse_adb.phi_z_pars[2]
    gamma = pulse_adb.mat.kappa * abs(pulse_adb.init_amp[idt, idp])
    gain = 1.0
    if (lbd_1 <= lbd <= lbd_2)
        gain = exp(pi*gamma^2 / dk_1)
    end
    
    return gain
end;

"""
phase_rsb_lbd
"""
function phase_rsb_lbd(lbd, pulse_adb::nl_pulse_adb; idt = 1, ids = 1, idi = 2, idp = 3, z0 = 0)
    (zc, dk_1) = pulse_adb.phi_z_pars;
    omega = (2*pi*c0) / (lbd * 1e12) * 1e6
    omega_sc = pulse_adb.mat.omega_c[ids];
    omega_ic = pulse_adb.mat.omega_c[idi];
    dbeta_si = beta_1_func(pulse_adb.mat, omega_sc, ids) - beta_1_func(pulse_adb.mat, omega_ic, idi);
    zpm = zc + (omega - omega_sc) / (dk_1) * dbeta_si;
    As = pulse_adb.init_amp[idt, idp];
    Aif = 1im * conj(As) * exp(-1im * dk_1/2 * zpm^2);
    phase = angle(Aif)
    
    return phase
end;

"""
Amplitude threshold for CW OPO (linearly poled)
"""
function cth_cw_opo_lin(loss, kappa, dk_1)
    return (1/kappa) * sqrt(dk_1/(2*pi) * log(1/(1-loss)))
end

end