"""
Pulse Module: Functions to generate and manipulate optical pulses\n
Updated: 6 Mar 2025\n

This module provides functions to generate various optical pulse shapes, including constant,
step, Gaussian, hyperbolic secant, random pulses, and complex random pulses. In addition,
it offers tools for simulating pulse dispersion under both second-order and full dispersion,
in the time and Fourier domains. It also includes routines for generating simulton solutions,
which are important in nonlinear optical processes.

Exported Functions and Structures:
  - Pulse shape functions: ff_const, ff_step, ff_gauss, ff_sech1, ff_sech, ff_rand, ff_gauss_rand, ff_rand_complex
        --> Typical form: ff_gauss(t, A, σ)
  - Pulse shape dictionary: pulse_func
  - Utility function: heaviside
  - Pulse parameter structure: pars_pulse
  - Pulse dispersion functions: pulse_disp, pulse_disp_FT, pulse_disp_FT_fd
  - Pulse generation: gen_pulse
  - Simulton parameters and pulses: simulton_pars, simulton_pulse, simulton_pulse_2
"""
module Pulse

using Random, Distributions
using Materials
export ff_const, ff_ste, ff_gauss, ff_sech1, ff_sech, ff_rand, ff_gauss_rand, ff_rand_complex
export heaviside, gen_pulse, pulse_func

ff_const(t, A, σ) = A
ff_step(t, A, σ) = A * (heaviside(σ/2 + t) + heaviside(σ/2 - t) - 1)
ff_gauss(t, A, σ) = A/(σ*sqrt(2*pi))*exp(-1/2*(t/σ)^2)
ff_sech1(t, A, τ) = A * sech(t/τ)
ff_sech(t, A, τ) = A * sech(t/τ)^2
ff_rand(t, A, σ) = A * (2 * rand() - 1)
ff_gauss_rand(t, A, σ) = A /(σ*sqrt(2*pi)) *exp(-1/2*(t/σ)^2) * (2 * rand() - 1)

function ff_rand_complex(t, A, σ)
    dist = Normal(0, A)
    return complex(rand(dist), rand(dist))
end

# Pulse dictionary
pulse_func = Dict("const" => ff_const, "rand" => ff_rand, "rand_complex" => ff_rand_complex,
    "gauss" => ff_gauss, "gauss_rand" => ff_gauss_rand, 
    "sech" => ff_sech, "sech1" => ff_sech1);

function heaviside(t)
   0.5 * (sign(t) + 1)
end;

export pars_pulse
"""
pars_pulse\n
Class for optical pulse\n

amp_max, T_pulse, N_div, f_pulse\n
amp_max: (N_pulse, )\n
omg_ctr: (N_pulse, )\n
T_pulse: (N_pulse, )\n
N_div: float\n
    Number of sampled points: 2^n
f_pulse: function OR Vector of functions\n
    The function defining the pulse shape
"""
mutable struct pars_pulse
    amp_max
    omg_ctr
    T_pulse
    N_div::Integer
    f_pulse
    pars_pulse() = new()
end

export pulse_disp, pulse_disp_FT, pulse_disp_FT_fd
"""
pulse_disp(T, z, A, σ, β0, β1, β2)
pulse under second order dispersion
"""
function pulse_disp(T, z, A, σ, β0, β1, β2)
    γp = exp(1im*β0*z)/sqrt(2*pi*(σ^2-1im*β2*z))
    fp = exp(-(T-β1*z)^2/(2*(σ^2-1im*β2*z)))
    return A * γp * fp
end;

"""
pulse_disp(T, z, pulse::pars_pulse, mat::TaylorMaterial, id_pulse::Integer)
pulse under second order dispersion
"""
function pulse_disp(T, z, pulse::pars_pulse, mat::TaylorMaterial, id_pulse::Integer; pulse_ref = 0)
    A = pulse.amp_max[id_pulse]
    σ = pulse.T_pulse[id_pulse]
    β0 = mat.beta_array[1,id_pulse]
    if pulse_ref == 0
        β1 = mat.beta_array[2,id_pulse]
    else
        β1 = mat.beta_array[2,id_pulse] - mat.beta_array[2,pulse_ref]
    end
    
    β2 = mat.beta_array[3,id_pulse]
    γp = exp(-1im*β0*z)/sqrt(2*pi*(σ^2+1im*β2*z))
    fp = exp(-(T-β1*z)^2/(2*(σ^2+1im*β2*z)))
    return A * γp * fp
end;

"""
pulse_disp_FT(ω, z, A, σ, β0, β1, β2)
pulse Fourier component under second order dispersion
"""
function pulse_disp_FT(ω, z, A, σ, β0, β1, β2)
    out = A/sqrt(2*pi) * exp(-1im*z*(β0 + β1*ω + 0.5*β2*ω^2)) * exp(-(σ*ω)^2/2)
    return out
end;

function pulse_disp_FT(ω, z, pulse::pars_pulse, mat::TaylorMaterial, id_pulse::Integer; pulse_ref = 0)
    A = pulse.amp_max[id_pulse]
    σ = pulse.T_pulse[id_pulse]
    β0 = mat.beta_array[1,id_pulse]
    if pulse_ref == 0
        β1 = mat.beta_array[2,id_pulse]
    else
        β1 = mat.beta_array[2,id_pulse] - mat.beta_array[2,pulse_ref]
    end
    
    β2 = mat.beta_array[3,id_pulse]
    out = A/sqrt(2*pi) * exp(-1im*z*(β0 + β1*ω + 0.5*β2*ω^2)) * exp(-(σ*ω)^2/2)
    return out
end;

"""
pulse_disp_FT_fd(ω, z, A, σ,  ωctr::Float64, Avec::Vector{Float64}, Bvec::Vector{Float64}; beta_f = beta_sellmeier)
pulse Fourier component under full dispersion
"""
function pulse_disp_FT_fd(ω, z, pulse::pars_pulse, mat::Material, id_pulse::Integer; pulse_ref = 0)
    A = pulse.amp_max[id_pulse]
    σ = pulse.T_pulse[id_pulse]
    ωctr = pulse.omg_ctr[id_pulse]
    if pulse_ref == 0
        beta = beta_func(mat, ω + ωctr)
    else
        beta_1_ref = beta_1_func(mat, pulse.omg_ctr[pulse_ref])
        beta = beta_func(mat, ω + ωctr) - beta_1_ref * ω 
    end        
    out = A/sqrt(2*pi) * exp(-1im*z*beta) * exp(-(σ*ω)^2/2)
    return out
end;

"""
gen_pulse(amp_max, arr_time, Tpulse; ff = ff_sech)

Generation of pulses\n
To prepare the initial condition\n
Inputs:\n

Outputs:\n

"""
function gen_pulse(amp_max, arr_time, Tpulse; ff = ff_sech, floattype = ComplexF64)
    out_array = Matrix{floattype}(undef, length(arr_time), length(amp_max))
    for i in 1:length(amp_max)
        if typeof(ff) == Vector{Function}
            func = ff[i]
            out_array[:,i] = [func(t, amp_max[i], Tpulse[i]) for t in arr_time]
        else
            fpulse(x) = ff(x, amp_max[i], Tpulse[i])
            out_array[:,i] = fpulse.(arr_time);
        end
    end
    return out_array
end;

# Calculating the simulton solution
"""
simulton_pars(pars_dopa, Ψmax)

Inputs:\n
pars_dopa\n
    Keys: "beta" (dispersion), "omega" (freq), "chi"\n
Ψmax: Maximum pump amplitude
"""
function simulton_pars(pars_dopa, Ψmax)
    βm = pars_dopa["beta"]
    ωm = pars_dopa["omega"]
    χm = pars_dopa["kappa"]

    k1_0, k1_1, k1_2 = βm[1,:]; 
    k2_0, k2_1, k2_2 = βm[2,:];

    # Spatial and Time Scale
    z0 = 1/abs(χm * Ψmax)
    t0 = sqrt(z0 * abs(k1_2))

    num = 2 * (k2_2 - 2*k1_2)/abs(k1_2)
    den = z0 * (k2_0 - 2*k1_0)
    κsq = den/num 
    Ts = sqrt(num/den) * t0

    θ1 = -2 * κsq * sign(k1_2)
    θ2 = z0 * (k2_0 - 2*k1_0) - 2 * κsq * (k2_2/abs(k1_2))

    # Signal
    φ0 = sqrt(18 * (κsq)^2 * (k2_2/k1_2))
    # Pump
    ψ0 = -1im * (φ0^2) * (abs(k1_2)/k2_2) / (6 * κsq)
    return Dict("z0" => z0, "t0" => t0, "Ts" => Ts, 
                "theta_1" => θ1, "theta_2" => θ2,
                "Pump" => ψ0, "Signal" => φ0)
end;

function simulton_pulse(t, z, ψmax, pars_sim)
    z0 = pars_sim["z0"];
    t0 = pars_sim["t0"];  Ts = pars_sim["Ts"];
    θ1 = pars_sim["theta_1"];  θ2 = pars_sim["theta_2"];
    signal = pars_sim["Signal"];  pump = pars_sim["Pump"];

    amp_signal = ψmax .* signal .* exp(1im * (θ1 * z / z0))
    amp_pump = ψmax .* pump .* exp(1im * (θ2 * z / z0))

    out = hcat(ff_sech.(t, amp_signal, Ts), ff_sech.(t, amp_pump, Ts))
    return out
end;

function simulton_pulse_2(t, z, Amax, χ, β2)
    T0 = sqrt(6 * abs(β2)/(χ * Amax))
    pulse = sech.( t./T0 ) .^ 2
    amp_signal = (Amax * exp(1im * (χ * Amax / 3) * z)) .* pulse 
    amp_pump = (1im * (Amax/2) * exp(1im * (2 * χ * Amax / 3) * z)) .* pulse 
    out = hcat(amp_signal, amp_pump)
    return out
end;
end