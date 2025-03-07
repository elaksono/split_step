"""
ECPO Module
Updated: 6 Mar 2025

This module implements functions and classes for simulating and analyzing the propagation of optical pulses 
in nonlinear optical systems, with a focus on the ECPO methodology. It provides routines for:

  • Propagating pulses through various media (e.g., waveguide and single-mode fiber (SMF)) 
    using split-step methods with nonlinear and dispersive effects.
  • Applying phase and amplitude modulation to model electro-optic modulation (EOM) effects.
  • Computing the separatrix condition (δ(ω) + ϕ(τ) = 0) to analyze phase dynamics within the system.
  • Supporting both CPU-based and GPU-accelerated implementations for enhanced performance.

Key Functions:
  - pulsed_ecpo_map: Performs pulse propagation over multiple round-trips using CPU routines.
  - pulsed_ecpo_map_p: A GPU-accelerated variant of the propagation routine.
  - separatrix: Computes the separatrix curves by finding zeros of the phase function, aiding in 
    the analysis of the system's stability and dynamics.

Dependencies:
  This module relies on several external and custom packages:
    • CUDA for GPU computing.
    • FFTW for Fourier transform operations.
    • JLD and ProgressMeter for data management and progress monitoring.
    • Custom modules such as Materials, NLO, Pulse, Modulation, SplitStep, and ECPOclass to define 
      the optical system's properties, propagation algorithms, and class structures.

Usage Example:
  using ECPO
  result = pulsed_ecpo_map(my_ecpo_instance, my_nonlinearity_operator; dz=0.1)

This module is intended for researchers and engineers involved in the simulation and analysis of complex 
optical systems, providing a robust framework for modeling nonlinear optical phenomena and pulse dynamics.
"""
module ECPO

using CUDA
using Materials
using NLO, NLOclass, Pulse, Modulation, SplitStep, SplitStepParallel, ECPOclass
using JLD, ProgressMeter
using FFTW
using Roots

export pulsed_ecpo_map, pulsed_ecpo_map_p

function pulsed_ecpo_map(pulse_ecpo::ecpo_class, op_nl; dz = -1.0, step_method = rk4_step_nl!, 
        p_fft = nothing, modulate = nothing, save_all = true, dir_save = nothing, with_z = false)
    opa = pulse_ecpo.ecpo_pulse
    if (dz > 1e-10) calculate_exp_beta_dz(opa, dz) end
    
    # Waveguide
    L_wg = pulse_ecpo.L_wg
    # SMF
    L_smf = pulse_ecpo.L_smf
    
    # Round-trip
    T_cav = pulse_ecpo.T_cav
    n_round = pulse_ecpo.n_round
    p_bar = Progress(n_round, "ECPO cycle...")
    
    # Saves
    n_saves = pulse_ecpo.n_saves
    save_folder = pulse_ecpo.save_folder
    if !isnothing(dir_save)
        save_folder = dir_save
    end
    file_label = save_folder * "/" * pulse_ecpo.save_label
    
    # Modulation - f(t, T, pars_pm/am...)
    f_phase_mod = pulse_ecpo.phase_mod_func
    f_amp_mod = pulse_ecpo.amp_mod_func
    pm = zeros(ComplexF64, size(opa.arr_time))
    am = zeros(size(opa.arr_time))
    if isnothing(modulate)
        pm .= exp.(1im .* [f_phase_mod(t, pulse_ecpo.T_mod, pulse_ecpo.pars_pm...) for t in opa.arr_time])
        am .= cos.([f_amp_mod(t, pulse_ecpo.T_mod, pulse_ecpo.pars_am...) for t in opa.arr_time])
    else
        pm = modulate[1]
        am = modulate[2]
    end
    
    # Modulation detuning
    dϕ_mod = angle(exp(1im * 2*pi * (T_cav/pulse_ecpo.T_mod)))
        
    # Incoupling
    # in_tot = reshape(pulse_ecpo.in_tab, (1,:)) # Matrix(1 x Nmodes)
    # in_tot = repeat(in_tot, length(opa.arr_time))
    
    # Outcoupling
    out_tot = reshape(pulse_ecpo.out_tab, (1,:)) # Matrix(1 x Nmodes)
    out_tot = repeat(out_tot, length(opa.arr_time))
    
    # Total loss
    α_tot = reshape(pulse_ecpo.α_tab, (1,:)) # Matrix(1 x Nmodes)
    α_tot = repeat(α_tot, length(opa.arr_time))
    
    # FOR DEBUG
    test_amp = zeros(ComplexF64, 3*n_round, size(opa.current_amp)...)
    out_amp = zeros(ComplexF64, size(opa.current_amp)...)
    
    save(file_label * "_time.jld", "time", opa.arr_time)
    save(file_label * "_omega_s.jld", "omega", opa.arr_omega[:,1] .+ pulse_ecpo.omega_vec[1])
    
    # Calculate init_amp, current_amp, out_amp
    for i in 1:n_round       
        # OPA
        opa.mat = pulse_ecpo.wg_mat
        
        test_amp[3*(i-1) + 1,:,:] .= opa.current_amp
        split_step_nl!(L_wg, opa, op_nl; p_fft = p_fft, step_method = step_method, with_z = with_z)
        test_amp[3*(i-1) + 2,:,:] .= opa.current_amp
        
        # EOM
        opa.current_amp[:,1] .= pm .* opa.current_amp[:,1]
        opa.current_amp[:,1] .= am .* opa.current_amp[:,1]
        
        # SMF
        opa.mat = pulse_ecpo.smf_mat
        opa.init_amp .= opa.current_amp
        split_step_nl!(L_smf, opa, op_nl; p_fft = p_fft, step_method = step_method)
        
        # Signal tap out + Residual pump
        opa.out_amp .= sqrt.(out_tot) .* opa.current_amp
        test_amp[3*(i-1) + 3,:,:] .= opa.out_amp
        if mod(i, n_saves) == 0
            save(file_label * "_T$(i)_out.jld", "Round", i, "amp", opa.out_amp)
        end
        
        # Remaining field
        opa.current_amp .= sqrt.(1 .- α_tot) .* opa.current_amp
        # test_amp[5*(i-1) + 5,:,:] .= opa.current_amp
        
        # Replenishing amplitude
        new_amp = gen_pulse(opa.pulse.amp_max, opa.arr_time, opa.pulse.T_pulse, ff=opa.pulse.f_pulse)
        opa.current_amp[:,3] .= new_amp[:,3]
        opa.current_amp[:,1] .= opa.current_amp[:,1] .+ (pulse_ecpo.in_tab[1] .* new_amp[:,1])
        opa.init_amp .= opa.current_amp
        
        # Including detuning in the modulation
        """
        pulse_ecpo.pars_pm[end] += dϕ_mod
        pulse_ecpo.pars_am[end] += dϕ_mod
        if isnothing(modulate) && (abs(dϕ_mod) > 1e-8)
            pm .= exp.(1im .* [f_phase_mod(t, pulse_ecpo.T_mod, pulse_ecpo.pars_pm...) for t in opa.arr_time])
            am .= cos.([f_amp_mod(t, pulse_ecpo.T_mod, pulse_ecpo.pars_am...) for t in opa.arr_time])
        end
        """
        next!(p_bar)
    end
    
    return test_amp
end;

# Calculating the separatrix
function f_ecpo_phase(lbd, t, pulse_ecpo::ecpo_class; fac = 1.0)
    omg = (3e8) * (1e6/lbd[1]) * 1e-12 * (2*pi)
    omg_c = pulse_ecpo.omega_vec[1]
    f_phase = pulse_ecpo.phase_mod_func
    ff = f_phase(t, pulse_ecpo.T_mod, pulse_ecpo.pars_pm...)
    gg = beta_func(pulse_ecpo.wg_mat, omg) * pulse_ecpo.L_wg
    gg_c = beta_func(pulse_ecpo.wg_mat, omg_c) * pulse_ecpo.L_wg
    gg_c += beta_1_func(pulse_ecpo.wg_mat, omg_c) * (omg - omg_c) * pulse_ecpo.L_wg
    return ff - fac * (gg - gg_c)
end;

export separatrix
"""
Calculating the separatrix: δ(ω) + ϕ(τ) = 0

Default:
separatrix(pulse_ecpo::ecpo; Nt = 50, u1 = 1.4, u2 = 1.7, t_symmetric = true, dir_save = nothing, save = true)
"""
function separatrix(pulse_ecpo::ecpo_class; Nt = 50, u1 = 1.4, u2 = 1.7, t_symmetric = true, 
        dir_save = nothing, save_data = true, fac = 1.0)   
    save_folder = pulse_ecpo.save_folder
    if !isnothing(dir_save)
        save_folder = dir_save
    end
    file_label = save_folder * "/" * "ecpo_spx.jld"
    
    if t_symmetric
        t_vec_p = range(0.0, stop = pulse_ecpo.T_mod/2, length = Nt + 1)
        branch_1_p = [find_zero(u -> f_ecpo_phase(u, t, pulse_ecpo; fac = fac), u1) for t in t_vec_p]
        branch_2_p = [find_zero(u -> f_ecpo_phase(u, t, pulse_ecpo; fac = fac), u2) for t in t_vec_p]
        t_vec = [ .-t_vec_p[end:-1:2]; t_vec_p]
        branch_1 = [ branch_2_p[end:-1:2]; branch_1_p ]
        branch_2 = [ branch_1_p[end:-1:2]; branch_2_p ]
    else
        t_vec = range(-pulse_ecpo.T_mod/2, stop = pulse_ecpo.T_mod/2, length = 2*Nt + 1)
        branch_1 = [find_zero(u -> f_ecpo_phase(u, t, pulse_ecpo; fac = fac), u1) for t in t_vec]
        branch_2 = [find_zero(u -> f_ecpo_phase(u, t, pulse_ecpo; fac = fac), u2) for t in t_vec]
    end
    
    output = Dict("time" => t_vec, "branch_1" => branch_1, "branch_2" => branch_2)
    if save_data
        save(file_label, "time", t_vec, "branch_1", branch_1, "branch_2", branch_2)
    end
    return output
end;

# IN-PROGRESS
function pulsed_ecpo_map_p(pulse_ecpo_p::ecpo_p, op_nl; dz = 0.1, step_method = rk4_step_p!,
        p_fft = nothing, modulate = nothing, save_all = true, dir_save = nothing)
    # Waveguide
    L_wg = pulse_ecpo_p.L_wg
    
    # Round-trip
    n_round = pulse_ecpo_p.n_round
    p_bar = Progress(n_round, "ECPO cycle...")
    
    # Saves
    n_saves = pulse_ecpo_p.n_saves
    save_folder = pulse_ecpo_p.ecpo_vec[1].save_folder
    if !isnothing(dir_save)
        save_folder = dir_save
    end
    save_label = save_folder * pulse_ecpo_p.file_label
    
    # losses
    in_tot_p = pulse_ecpo_p.in_tot_p
    out_tot_p = pulse_ecpo_p.out_tot_p
    α_tot_p = pulse_ecpo_p.α_tot_p
    
    # Modulation
    pm_p = pulse_ecpo_p.pm_p
    am_p = pulse_ecpo_p.am_p
    
    # Pulse::nl_pulse_p
    pulse_p = pulse_ecpo_p.ecpo_pulse_p
    sync_init_amp_p = CuArray(zeros(ComplexF32, size(pulse_p.init_amp_p)))
    sync_init_amp_p .= pulse_p.init_amp_p
    
    for i in 1:n_round
        split_step_p!(L_wg, pulse_ecpo_p.ecpo_pulse_p, op_nl, dz = dz, p_fft = p_fft, step_method = step_method) # NL
        
        # Modulating the signal
        pulse_p.current_amp_p .= pm_p .* pulse_p.current_amp_p
        pulse_p.current_amp_p .= am_p .* pulse_p.current_amp_p
        
        # Signal tap out
        pulse_p.out_amp_p .= sqrt.(out_tot_p) .* pulse_p.current_amp_p
        
        if mod(i, n_saves) == 0
            save(save_label * "_t$(i).jld", "Time", i, "amp", Array(pulse_p.out_amp_p))
        end
        
        # Remaining field
        pulse_p.current_amp_p .= sqrt.(1 .- α_tot_p) .* pulse_p.current_amp_p
        
        # Current_amp replenished for next cycle
        pulse_p.current_amp_p .= pulse_p.current_amp_p .+ sqrt.(in_tot_p) .* sync_init_amp_p
        pulse_p.init_amp_p .= pulse_p.current_amp_p
        
        next!(p_bar)
    end
    
    return
end;

end