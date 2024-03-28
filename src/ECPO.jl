module ECPO

using CUDA
using NLO, NLOclass, Pulse, Modulation, SplitStep, SplitStepParallel, ECPOclass
using JLD, ProgressMeter
export pulsed_ecpo_map, pulsed_ecpo_map_p

function pulsed_ecpo_map(pulse_ecpo::ecpo, op_nl; dz = 0.1, step_method = rk4_step_nl!, 
        p_fft = nothing, modulate = nothing, save_all = true, dir_save = nothing)
    pulse = pulse_ecpo.ecpo_pulse
    
    # Waveguide
    L_wg = pulse_ecpo.L_wg
    
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
    file_label = save_folder * pulse_ecpo.save_label
    
    # Modulation - f(t, T, pars_pm/am...)
    f_phase_mod = pulse_ecpo.phase_mod_func
    f_amp_mod = pulse_ecpo.amp_mod_func
    pm = zeros(ComplexF64, size(pulse.arr_time))
    am = zeros(size(pulse.arr_time))
    if isnothing(modulate)
        pm .= exp.(-1im .* [f_phase_mod(t, pulse_ecpo.T_mod, pulse_ecpo.pars_pm[1], pulse_ecpo.pars_pm[2]) for t in pulse.arr_time])
        am .= cos.([f_amp_mod(t, pulse_ecpo.T_mod, pulse_ecpo.pars_am[1], pulse_ecpo.pars_am[2]) for t in pulse.arr_time])
    else
        pm = modulate[1]
        am = modulate[2]
    end
    
    dT_mod = (mod(T_cav, pulse_ecpo.T_mod), mod(T_cav, pulse_ecpo.T_mod) - pulse_ecpo.T_mod)
    min_id = argmin(abs.(dT_mod))
    dT_mod = dT_mod[min_id]
    dϕ_mod = 2*pi * (dT_mod/pulse_ecpo.T_mod)
    
    # Injected amp
    sync_init_amp = zeros(ComplexF64, size(pulse.init_amp))
    sync_init_amp .= pulse.init_amp
    
    # Incoupling
    in_tot = reshape(pulse_ecpo.in_tab, (1,:)) # Matrix(1 x Nmodes)
    in_tot = repeat(in_tot, length(pulse.arr_time))
    
    # Outcoupling
    out_tot = reshape(pulse_ecpo.out_tab, (1,:)) # Matrix(1 x Nmodes)
    out_tot = repeat(out_tot, length(pulse.arr_time))
    
    # Total loss
    α_tot = reshape(pulse_ecpo.α_tab, (1,:)) # Matrix(1 x Nmodes)
    α_tot = repeat(α_tot, length(pulse.arr_time))
    
    # Calculate init_amp, current_amp, out_amp
    for i in 1:n_round
        split_step_nl!(L_wg, pulse, op_nl, dz = dz, p_fft = p_fft, step_method = step_method) # NL
        
        # Modulating the signal
        pulse.current_amp .= pm .* pulse.current_amp
        pulse.current_amp .= am .* pulse.current_amp
        
        # Signal tap out
        pulse.out_amp .= sqrt.(out_tot) .* pulse.current_amp
        if mod(i, n_saves) == 0
            save(file_label * "_t$(i).jld", "Time", i, "amp", pulse.out_amp)
        end
        
        # Remaining field
        pulse.current_amp .= sqrt.(1 .- α_tot) .* pulse.current_amp
        
        # Current_amp replenished for next cycle
        pulse.current_amp .= pulse.current_amp .+ sqrt.(in_tot) .* sync_init_amp
        pulse.init_amp .= pulse.current_amp
        
        # Next step modulation
        if isnothing(modulate) && (abs(dϕ_mod) > 1e-10)
            pm .= exp.(-1im .* [f_phase_mod(t, pulse_ecpo.T_mod, pulse_ecpo.pars_pm[1], 
                        pulse_ecpo.pars_pm[2] + i * dϕ_mod) for t in pulse.arr_time])
            am .= cos.([f_amp_mod(t, pulse_ecpo.T_mod, pulse_ecpo.pars_am[1], 
                        pulse_ecpo.pars_am[2] + i * dϕ_mod) for t in pulse.arr_time])
        end
        
        next!(p_bar)
    end
    
    return
end;

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