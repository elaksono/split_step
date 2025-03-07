"""
NLOclass.jl - Nonlinear Optical Simulation Classes\n
Updated: 6 March 2025\n

This module provides the essential class definitions and routines for simulating nonlinear optical 
pulse propagation. It integrates various components including pulse generation, material dispersion, 
and nonlinear interactions. The module depends on several other modules:
  - Pulse: For generating various optical pulse shapes.
  - Constant: For physical constants.
  - CUDA and FFTW: For GPU and fast Fourier transform operations.
  - Materials: For material properties and dispersion functions.
  - Modulation: For modulation-related functions.

Key Components:\n

1. nl_pulse Mutable Structure:
   - Encapsulates the simulation parameters and state for nonlinear optical pulse propagation.
   - Fields include:
     • mat::Material: The material properties (e.g., dispersion coefficients).
     • pulse::pars_pulse: Pulse parameters such as amplitude, central frequency, and pulse duration.
     • dz::Real: The fixed propagation step size.
     • Tmax, dT, arr_time: Parameters defining the time grid (maximum time window, time step, and array of time samples).
     • init_amp, current_amp, out_amp, temp_amp: Arrays to store the initial, current, and output amplitudes, and temporary storage.
     • κ_matrix: Matrix for the nonlinear coupling coefficients.
     • nl_func: Function that defines the nonlinear phase accumulation (typically of the form exp(1im * phi)).
     • phi_z: Function (or function handle) that specifies the phase evolution along the propagation direction.
     • Frequency-domain fields (arr_omega, beta_ω, exp_beta_dz, dk_qpm, delta_k) for the simulation of spectral dynamics.

2. nl_pulse Constructor:
   - The constructor initializes a nl_pulse instance given a Material object and pulse parameters.
   - Key steps in the constructor:
     • Determines the maximum time window (Tmax) for the simulation based on the provided pulse durations and an optional scaling factor.
     • Computes the time step (dT) and constructs the time array (arr_time) by concatenating positive and negative time intervals.
     • Generates the initial amplitude (init_amp) using the gen_pulse function from the Pulse module.
     • Accepts several keyword arguments to control simulation aspects such as phase mismatch, frequency resolution (d_omega), and the nonlinear function (nl_func).
     • Supports options for quasi-phase matching adjustments via parameters like phase_mismatch and delta_k.
     
Usage:\n
   The nl_pulse structure serves as the central data container for simulating nonlinear optical phenomena,
   where the evolution of the pulse is computed in both time and frequency domains under the influence of 
   material dispersion and nonlinear effects.\n
"""
module NLOclass

using Pulse
using Constant
using CUDA, FFTW
using Materials
using Modulation

export nl_pulse, nl_pulse_adb
export nl_pulse_p
export calculate_exp_beta_dz
export nl_class

"""
nl_pulse\n
AN ESSENTIAL CLASS for the simulation

"""
mutable struct nl_pulse
    mat::Material
    pulse::pars_pulse

    # Fixed step
    dz::Real
    
    # Time bins
    Tmax::Real
    dT::Real
    arr_time::Vector{Real}

    # Amplitudes
    init_amp
    current_amp
    out_amp
    temp_amp # Cache
    
    # Nonlinearity
    κ_matrix
    nl_func
    phi_z
    
    # Frequency bins
    # exp_ω0T
    arr_omega
    beta_ω
    exp_beta_dz
    dk_qpm
    delta_k
end

"""
nl_pulse(mat::Material, pulse::pars_pulse; dz = 1e-2, Tm = 0.0, fac = 10, pulse_ref = 0, 
        d_omega = 1e-3, calculate_dk = false,
        zero_beta0 = true, 
        phase_mismatch = 0.0, delta_k = nothing, nl_func = phi -> exp(1im * phi), 
        phi_z_func = phi_z_con, phi_z_pars = ())\n
\n
In this version, the phase_mismatch is enforced at the omg_ctr. It is defined w.r.t dK = beta_p - beta_s - beta_i
    => delta_k = dK * phase_mismatch
"""
function nl_pulse(mat::Material, pulse::pars_pulse; dz = 1e-2, Tm = 0.0, fac = 10, pulse_ref = 0, 
        d_omega = 1e-3, calculate_dk = false,
        zero_beta0 = true, 
        phase_mismatch = 0.0, delta_k = nothing, nl_func = phi -> exp(1im * phi), 
        phi_z_func = phi_z_con, phi_z_pars = ())
    Tmax = Tm > maximum(pulse.T_pulse) ? Tm : fac * maximum(pulse.T_pulse)
    dT = 2*Tmax / (pulse.N_div)
    arr_time = [collect(0:dT:Tmax-dT/2); collect(-Tmax:dT:-dT/2)]

    # Amplitude
    N_pulse = length(pulse.amp_max)
    init_amp = gen_pulse(pulse.amp_max, arr_time, pulse.T_pulse, ff=pulse.f_pulse)
    current_amp = copy(init_amp)
    out_amp = zeros(typeof(init_amp[1,1]), size(init_amp)...)
    temp_amp = zeros(typeof(init_amp[1,1]), size(init_amp)..., 5)    

    # Nonlinearity
    κ_matrix = (mat.kappa) .* ones(size(init_amp)[1])
    
    # Frequency
    # exp_ω0T = ones(ComplexF32, size(current_amp)...)
    arr_omega = zeros(size(current_amp)...)
    beta_ω = zeros(size(current_amp)...)
    exp_beta_dz = zeros(ComplexF64, size(current_amp)...)
    
    if calculate_dk
        if typeof(mat) <: TaylorMaterial
            dk_qpm = (beta_func(mat, 0.0, 3) - beta_func(mat, 0.0, 1) - beta_func(mat, 0.0, 2))
        else
            dk_qpm = (beta_func(mat, pulse.omg_ctr[3]) - beta_func(mat, pulse.omg_ctr[1]) - beta_func(mat, pulse.omg_ctr[2]))
        end
    else
        dk_qpm = nothing
    end
    
    
    if pulse_ref > 0
        if typeof(mat) <: TaylorMaterial
            beta_1_ref = (beta_func(mat, d_omega/2, pulse_ref) - beta_func(mat, -d_omega/2, pulse_ref)) / d_omega
        else
            beta_1_ref = beta_1_func(mat, pulse.omg_ctr[pulse_ref]; d_omega = d_omega)
        end
    end
        
    for i in 1:N_pulse
        arr_omega[:,i] .= fftfreq(length(arr_time)) .* (2*pi/dT)
        
        if typeof(mat) <: TaylorMaterial
            beta_ω[:,i] .= [beta_func(mat, omg, i) for omg in arr_omega[:,i]]
            if zero_beta0
                beta_ω[:,i] .= beta_ω[:,i] .- beta_func(mat, 0.0, i)
            end
        else
            beta_ω[:,i] .= [beta_func(mat, omg + pulse.omg_ctr[i]) for omg in arr_omega[:,i]]
            if zero_beta0
                beta_ω[:,i] .= beta_ω[:,i] .- beta_func(mat, pulse.omg_ctr[i])
            end
        end
        
        
        if pulse_ref > 0
            beta_ω[:,i] .= beta_ω[:,i] .- (beta_1_ref .* arr_omega[:,i])
        end
        
        exp_beta_dz[:,i] .= exp.((-1im * dz) .* beta_ω[:,i])
    end
    
    if isnothing(delta_k)
        delta_k = phase_mismatch * dk_qpm
    end
    phi_z = z -> phi_z_func(z, delta_k, phi_z_pars...)

    nl_pulse(mat, pulse, dz,
        Tmax, dT, arr_time,
        init_amp, current_amp, out_amp, temp_amp, 
        κ_matrix, nl_func, phi_z,
        arr_omega, beta_ω, exp_beta_dz, dk_qpm, delta_k)
end;

"""
nl_pulse_adb\n
A class for adiabatically poled waveguide simulation
"""
mutable struct nl_pulse_adb
    mat::Material
    pulse::pars_pulse

    # Fixed step
    dz::Real
    
    # Time bins
    Tmax::Real
    dT::Real
    arr_time::Vector{Real}

    # Amplitudes
    init_amp
    current_amp
    out_amp
    temp_amp # Cache
    
    # Nonlinearity
    κ_matrix
    nl_func
    phi_z
    g_func
    
    # Frequency bins
    # exp_ω0T
    arr_omega
    beta_ω
    exp_beta_dz
    dk_qpm
    delta_k
end

"""
nl_pulse_adb(mat::Material, pulse::pars_pulse; dz = 1e-2, Tm = 0.0, fac = 10, pulse_ref = 0, 
        d_omega = 1e-3, calculate_dk = false,
        zero_beta0 = true, 
        qpm_lambda = 20.0, delta_k = nothing, nl_func = phi -> exp(1im * phi), 
        phi_z_func = phi_z_con, phi_z_pars = ())\n
\n
In this version, we assume a quasi-phase matching point for central frequency defined by qpm_lambda\n
i.e., dK = beta_p - beta_s - beta_i\n
=> delta_k = dK - 2*pi/qpm_lambda
"""
function nl_pulse_adb(mat::Material, pulse::pars_pulse; dz = 1e-2, Tm = 0.0, fac = 10, pulse_ref = 0, 
        d_omega = 1e-3, calculate_dk = false,
        zero_beta0 = true, 
        qpm_lambda = 20.0, delta_k = nothing, nl_func = phi -> exp(1im * phi), 
        phi_z_func = phi_z_con, phi_z_pars = (), g_func = g_con, g_pars = ())
    Tmax = Tm > maximum(pulse.T_pulse) ? Tm : fac * maximum(pulse.T_pulse)
    dT = 2*Tmax / (pulse.N_div)
    arr_time = [collect(0:dT:Tmax-dT/2); collect(-Tmax:dT:-dT/2)]

    # Amplitude
    N_pulse = length(pulse.amp_max)
    init_amp = gen_pulse(pulse.amp_max, arr_time, pulse.T_pulse, ff=pulse.f_pulse)
    current_amp = copy(init_amp)
    out_amp = zeros(typeof(init_amp[1,1]), size(init_amp)...)
    temp_amp = zeros(typeof(init_amp[1,1]), size(init_amp)..., 5)    

    # Nonlinearity
    κ_matrix = (mat.kappa) .* ones(size(init_amp)[1])
    
    # Frequency
    # exp_ω0T = ones(ComplexF32, size(current_amp)...)
    arr_omega = zeros(size(current_amp)...)
    beta_ω = zeros(size(current_amp)...)
    exp_beta_dz = zeros(ComplexF64, size(current_amp)...)
    
    if calculate_dk
        if typeof(mat) <: TaylorMaterial
            dk_qpm = (beta_func(mat, 0.0, 3) - beta_func(mat, 0.0, 1) - beta_func(mat, 0.0, 2))
        else
            dk_qpm = (beta_func(mat, pulse.omg_ctr[3]) - beta_func(mat, pulse.omg_ctr[1]) - beta_func(mat, pulse.omg_ctr[2]))
        end
    else
        dk_qpm = nothing
    end
    
    
    if pulse_ref > 0
        if typeof(mat) <: TaylorMaterial
            beta_1_ref = (beta_func(mat, d_omega/2, pulse_ref) - beta_func(mat, -d_omega/2, pulse_ref)) / d_omega
        else
            beta_1_ref = beta_1_func(mat, pulse.omg_ctr[pulse_ref]; d_omega = d_omega)
        end
    end
        
    for i in 1:N_pulse
        arr_omega[:,i] .= fftfreq(length(arr_time)) .* (2*pi/dT)
        
        if typeof(mat) <: TaylorMaterial
            beta_ω[:,i] .= [beta_func(mat, omg, i) for omg in arr_omega[:,i]]
            if zero_beta0
                beta_ω[:,i] .= beta_ω[:,i] .- beta_func(mat, 0.0, i)
            end
        else
            beta_ω[:,i] .= [beta_func(mat, omg + pulse.omg_ctr[i]) for omg in arr_omega[:,i]]
            if zero_beta0
                beta_ω[:,i] .= beta_ω[:,i] .- beta_func(mat, pulse.omg_ctr[i])
            end
        end
        
        
        if pulse_ref > 0
            beta_ω[:,i] .= beta_ω[:,i] .- (beta_1_ref .* arr_omega[:,i])
        end
        
        exp_beta_dz[:,i] .= exp.((-1im * dz) .* beta_ω[:,i])
    end
    
    if isnothing(delta_k)
        delta_k = dk_qpm - (2*pi)/(qpm_lambda * 1e-3)
    end
    
    phi_z = z -> phi_z_func(z, delta_k, phi_z_pars...)
    g_func = z -> g_func(z, g_pars...)

    nl_pulse_adb(mat, pulse, dz,
        Tmax, dT, arr_time,
        init_amp, current_amp, out_amp, temp_amp, 
        κ_matrix, nl_func, phi_z, g_func,
        arr_omega, beta_ω, exp_beta_dz, dk_qpm, delta_k)
end;

nl_class = Union{nl_pulse, nl_pulse_adb}

function calculate_exp_beta_dz(pulse::nl_class, dz)
    pulse.dz = dz
    pulse.exp_beta_dz = exp.((-1im * dz) .* pulse.beta_ω) 
end

"""
IN-PROGRESS
"""
# STRUCT FOR GPU
mutable struct nl_pulse_p
    nl_vec::Vector{nl_pulse}
    
    # SOMETHING IS MISSING HERE... TIME
    
    # Amplitudes
    init_amp_p::CuArray
    current_amp_p::CuArray
    out_amp_p::CuArray
    temp_amp_p::CuArray
    
    # Nonlinearity matrix
    κ_matrix_p::CuArray

    # Frequency bins
    arr_omega_p::CuArray
    exp_ω0T_p::CuArray
    beta_ω_p::CuArray
end

function nl_pulse_p(nl_vec::Vector{nl_pulse})
    n_cases = length(nl_vec)
    n_bins = size(nl_vec[begin].init_amp)[1]
    n_modes = size(nl_vec[begin].init_amp)[2]
    
    # Initiate amplitudes
    # Dimension: N(time bins x parameters cases x modes)
    init_amp_p = CUDA.zeros(ComplexF32, n_bins, n_cases, n_modes)
    current_amp_p = CUDA.zeros(ComplexF32, n_bins, n_cases, n_modes)
    out_amp_p = CUDA.zeros(ComplexF32, n_bins, n_cases, n_modes)
    temp_amp_p = CUDA.zeros(ComplexF32, n_bins, n_cases, n_modes, 5)
    
    # === TO DO HERE ===
    for (c,m) in Iterators.product(1:n_cases, 1:n_modes)
        init_amp_p[:,c,m] .= CuArray(nl_vec[c].init_amp[:,m])
    end
    
    current_amp_p .= init_amp_p
    
    # Initiate nonlinear coefficients
    κ_matrix_p = CUDA.zeros(Float32, n_bins, n_cases)
    for c in 1:n_cases
        κ_matrix_p[:,c] .= nl_vec[c].nl.κnl .* CUDA.ones(n_bins)
    end
    
    arr_omega_p = CUDA.zeros(Float32, size(current_amp_p)...)
    exp_ω0T_p = CUDA.zeros(ComplexF32, size(current_amp_p)...)
    beta_ω_p = CUDA.zeros(Float32, size(current_amp_p)...)

    for (c,m) in Iterators.product(1:n_cases, 1:n_modes)
        dT = nl_vec[c].dT
        arr_time = nl_vec[c].arr_time;
        arr_omega_p[:,c,m] .= CuArray(fftfreq(size(arr_omega_p)[1]) .* (2*pi/dT))
        arr_omega_p[:,c,m] .= (nl_vec[c].nl.ωnl[m] .- arr_omega_p[:,c,m])
        exp_ω0T_p[:,c,m] .= CuArray(exp.((1im * nl_vec[c].nl.ωnl[m]) .* arr_time))
        beta_ω_p[:,c,m] .= CuArray(f_taylor(arr_omega_p[:,c,m], nl_vec[c].nl.βnl[:,m]))
    end
    
    nl_pulse_p(nl_vec, init_amp_p, current_amp_p, out_amp_p, temp_amp_p, 
               κ_matrix_p, arr_omega_p, exp_ω0T_p, beta_ω_p)
end;

function refresh_nl_pulse_p!(dpulse_p::nl_pulse_p, nl_vec::Vector{nl_pulse})
    for (c,m) in Iterators.product(1:n_cases, 1:n_modes)
        dpulse_p.current_amp_p[:,c,m] .= CuArray(nl_vec[c].init_amp[:,m])
    end
    return
end

end