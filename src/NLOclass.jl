module NLOclass

using Pulse: gen_pulse
using CUDA, FFTW
export f_taylor, pars_nl, pars_pulse, nl_pulse, nl_pulse_p

"""
Taylor series: useful for dispersion
"""
function f_taylor(x, arr_γ)
    return sum(arr_γ .* [x.^j ./ factorial(j) for j in collect(0:length(arr_γ)-1)])
end;

mutable struct pars_nl
    βnl
    ωnl
    κnl::Real
    pars_nl() = new()
end

mutable struct pars_pulse
    amp_max
    T_pulse
    N_div::Integer
    f_pulse

    pars_pulse() = new()
end

mutable struct nl_pulse
    nl::pars_nl
    pulse::pars_pulse

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
    
    # Frequency bins
    arr_omega
    exp_ω0T
    beta_ω
end

function nl_pulse(nl::pars_nl, pulse::pars_pulse; fac = 10)
    # Time
    Tmax = fac * maximum(pulse.T_pulse)
    dT = 2*Tmax / (pulse.N_div)
    
    # We want to follow the FFT definition
    arr_time = collect(0:dT:Tmax-dT/2)
    append!(arr_time, collect(-Tmax:dT:-dT/2));

    # Amplitude
    init_amp = gen_pulse(pulse.amp_max, arr_time, pulse.T_pulse, ff=pulse.f_pulse)
    current_amp = copy(init_amp)
    out_amp = zeros(typeof(init_amp[1,1]), size(init_amp)...)
    temp_amp = zeros(typeof(init_amp[1,1]), size(init_amp)..., 5)    

    # Nonlinearity
    κ_matrix = (nl.κnl) .* ones(size(init_amp)[1])
    
    # Frequency
    arr_omega = zeros(size(current_amp)...)
    exp_ω0T = zeros(ComplexF32, size(current_amp)...)
    beta_ω = zeros(size(current_amp)...)

    for i in 1:length(nl.ωnl)
        arr_omega[:,i] .= fftfreq(length(arr_time)) .* (2*pi/dT)
        arr_omega[:,i] .= (nl.ωnl[i] .- arr_omega[:,i])
        exp_ω0T[:,i] .= exp.((1im * nl.ωnl[i]) .* arr_time)
        beta_ω[:,i] .= f_taylor(arr_omega[:,i], nl.βnl[:,i])
    end

    nl_pulse(nl, pulse, 
        Tmax, dT, arr_time,
        init_amp, current_amp, out_amp, temp_amp, 
        κ_matrix, arr_omega, exp_ω0T, beta_ω)
end;

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