module Pulse
using Random
export ff_const, ff_ste, ff_gauss, ff_sech1, ff_sech, ff_rand, ff_gauss_rand, heaviside, gen_pulse

ff_const(t, A, σ) = A
ff_step(t, A, σ) = A * (heaviside(σ/2 + t) + heaviside(σ/2 - t) - 1)
ff_gauss(t, A, σ) = A/(σ*sqrt(2*pi))*exp(-1/2*(t/σ)^2)
ff_sech1(t, A, τ) = A * sech(t/τ)
ff_sech(t, A, τ) = A * sech(t/τ)^2
ff_rand(t, A, σ) = A * (2 * rand() - 1)
ff_gauss_rand(t, A, σ) = A /(σ*sqrt(2*pi)) *exp(-1/2*(t/σ)^2) * (2 * rand() - 1)

function heaviside(t)
   0.5 * (sign(t) + 1)
end;

# To prepare the initial condition
"""
gen_pulse(amp_max, arr_time, Tpulse; ff = ff_sech)

Generation of pulses\n
Inputs:\n

Outputs:\n

"""
function gen_pulse(amp_max, arr_time, Tpulse; ff = ff_sech)
    out_array = Matrix{ComplexF32}(undef, length(arr_time), length(amp_max))
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