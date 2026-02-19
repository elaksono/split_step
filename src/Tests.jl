module Tests

using CSV, DataFrames, JSON, JLD, OrderedCollections, NPZ
using FFTW, GLM, LinearAlgebra, Random
using LaTeXStrings, ColorSchemes, Plots
using CUDA, FileIO

using DifferentialEquations
import Elliptic.Jacobi
using Roots

using Constant, Basic
using Materials
using Pulse, Modulation, Analytics
using NLOclass, NLO
using SplitStep

# Dispersion only
export opa_dispersion_only, plot_opa_dispersion_only, plot_opa_ft_dispersion_only
function opa_dispersion_only(disp::nl_class; step_method = rk4_step_nl!, 
        pulse_amp_vs_t = nothing, pulse_amp_vs_f = pulse_disp_FT, pulse_ref = 0, zero_beta0 = true)
    # Numerical solution
    p_fft! = plan_fft!(disp.init_amp, 1)
    p_ifft! = plan_ifft!(disp.init_amp, 1)
    Lz = disp.mat.prop.ll
    split_step_nl!(Lz, disp, op_nl_chi2!; step_method = step_method, p_fft = (p_fft!, p_ifft!))

    # Analytical solution
    sorted_time = sort(disp.arr_time)
    sorted_id = sortperm(disp.arr_time)

    arr_omega = (2*pi) .* fftfreq(disp.pulse.N_div) * (disp.pulse.N_div / (2*disp.Tmax))
    sorted_omega = sort(arr_omega) ./ (2*pi)
    sorted_f_id = sortperm(arr_omega)
    γf = sqrt(2*pi) * disp.pulse.N_div / (2 * disp.Tmax)

    # Amplitude vs omega
    sol_signal_ft = [γf .* pulse_amp_vs_f(ω, Lz, disp.pulse, disp.mat, 1; pulse_ref = pulse_ref, zero_beta0 = zero_beta0) 
        for ω in arr_omega]
    sol_idler_ft = [γf .* pulse_amp_vs_f(ω, Lz, disp.pulse, disp.mat, 2; pulse_ref = pulse_ref, zero_beta0 = zero_beta0) 
        for ω in arr_omega]
    sol_pump_ft = [γf .* pulse_amp_vs_f(ω, Lz, disp.pulse, disp.mat, 3; pulse_ref = pulse_ref, zero_beta0 = zero_beta0) 
        for ω in arr_omega]

    # Amplitude vs time
    if isnothing(pulse_amp_vs_t)
        sol_signal = ifft(sol_signal_ft)[sorted_id]
        sol_idler = ifft(sol_idler_ft)[sorted_id]
        sol_pump = ifft(sol_pump_ft)[sorted_id]
    else
        sol_signal = [pulse_amp_vs_t(t, Lz, disp.pulse, disp.mat, 1; pulse_ref = pulse_ref, zero_beta0 = zero_beta0) for t in sorted_time]
        sol_idler = [pulse_amp_vs_t(t, Lz, disp.pulse, disp.mat, 2; pulse_ref = pulse_ref, zero_beta0 = zero_beta0) for t in sorted_time]
        sol_pump = [pulse_amp_vs_t(t, Lz, disp.pulse, disp.mat, 3; pulse_ref = pulse_ref, zero_beta0 = zero_beta0) for t in sorted_time]
    end
    
    out_signal = disp.out_amp[sorted_id,1]
    out_idler = disp.out_amp[sorted_id,2]
    out_pump = disp.out_amp[sorted_id,3]
    
    sol_signal_ft = sol_signal_ft[sorted_f_id]
    sol_idler_ft = sol_idler_ft[sorted_f_id]
    sol_pump_ft = sol_pump_ft[sorted_f_id]
    out_signal_ft = fft(disp.out_amp[:,1])[sorted_f_id]
    out_idler_ft = fft(disp.out_amp[:,2])[sorted_f_id]
    out_pump_ft = fft(disp.out_amp[:,3])[sorted_f_id]
    
    return (
        arr_time      = disp.arr_time,
        sorted_time   = sorted_time,
        arr_omega     = arr_omega,
        sorted_omega   = sorted_omega,
        sol_signal_ft = sol_signal_ft,
        sol_idler_ft  = sol_idler_ft,
        sol_pump_ft   = sol_pump_ft,
        sol_signal    = sol_signal,
        sol_idler     = sol_idler,
        sol_pump      = sol_pump,
        out_signal_ft = out_signal_ft,
        out_idler_ft  = out_idler_ft,
        out_pump_ft   = out_pump_ft,
        out_signal    = out_signal,
        out_idler     = out_idler,
        out_pump      = out_pump
    )
end

function plot_opa_dispersion_only(output::NamedTuple, id_pulse::Integer; savefile = true, 
        filename = "./figures/disp_time_plot.png", mat_type = "Taylor", colors = [:blue, :green, :red, :cyan], 
        labels = ["Split Step - ", "Analytic - "], disp_plot = nothing)
    # Propagation in time domain
    if isnothing(disp_plot)
        disp_plot = plot(frame=:box, xlabel = L"Time $\tau$ (ps)", ylabel = L"Amplitude $A(\tau)$", legend=:bottomleft)
    end

    time_0 = output.sorted_time
    if id_pulse == 1
        for_plot_0 = (output.out_signal, output.sol_signal)
    elseif id_pulse == 2
        for_plot_0 = (output.out_idler, output.sol_idler)
    elseif id_pulse == 3
        for_plot_0 = (output.out_pump, output.sol_pump)
    end

    # Propagated pulse with second-order dispersion
    for j in 1:2
        plot!(disp_plot, time_0, real.(for_plot_0[j]), color=colors[j], label = labels[j]*mat_type)
        plot!(disp_plot, time_0, imag.(for_plot_0[j]), color=colors[j], line=:dash, label=:none)
    end
    
    if savefile
        savefig(filename)
    end
    
    disp_plot
    return disp_plot
end

function plot_opa_ft_dispersion_only(output::NamedTuple, id_pulse::Integer; savefile = true, 
        filename = "./figures/disp_omega_plot.png", mat_type = "Taylor", colors = [:blue, :green, :red, :cyan], 
        labels = ["Split Step - ", "Analytic - "], disp_plot = nothing, xlim = (-5,5))
    # Propagation in freq domain
    if isnothing(disp_plot)
        disp_plot = plot(frame=:box, xlim = xlim, xlabel = L"$\omega$ (THz)", ylabel = L"Amplitude $\tilde{A}(\omega)$")
    end

    omega_0 = output.sorted_omega
    if id_pulse == 1
        for_plot_0 = (output.out_signal_ft, output.sol_signal_ft)
    elseif id_pulse == 2
        for_plot_0 = (output.out_idler_ft, output.sol_idler_ft)
    elseif id_pulse == 3
        for_plot_0 = (output.out_pump_ft, output.sol_pump_ft)
    end

    # Propagated pulse with second-order dispersion
    for j in 1:2
        plot!(disp_plot, omega_0, real.(for_plot_0[j]), color=colors[j], label = labels[j]*mat_type)
        plot!(disp_plot, omega_0, imag.(for_plot_0[j]), color=colors[j], line=:dash, label=:none)
    end
    
    if savefile
        savefig(filename)
    end
    
    disp_plot
    return disp_plot
end


export dopa_simulton, plot_simulton

function dopa_simulton(sml::nl_class; step_method = rk4_step_nl!)
    p_fft! = plan_fft!(sml.init_amp, 1);
    p_ifft! = plan_ifft!(sml.init_amp, 1);
    Lz = sml.mat.prop.ll
    kappa = sml.mat.kappa
    beta_2 = sml.mat.beta_array[3,1]
    Amax = sml.pulse.amp_max[1]
    
    # Split-step simulation for simulton
    split_step_nl!(Lz, sml, op_nl_dopa!, step_method = rk4_step_nl!, p_fft = (p_fft!, p_ifft!))

    # Analytical result for simulton
    sol_sml = Pulse.simulton_pulse_2(sml.arr_time, Lz, Amax, kappa, beta_2);
    
    arr_time = sml.arr_time
    sorted_time = sort(sml.arr_time)
    sorted_id = sortperm(sml.arr_time)
    sol_sml = sol_sml[sorted_id,:] # Analytics
    out_sml = sml.out_amp[sorted_id,:] # Numerics
    
    return (
        arr_time = arr_time,
        sorted_time = sorted_time,
        sol_sml = sol_sml,
        out_sml = out_sml
    )
end

"""
plotCompare(arr_x, num_pulse, anl_pulse, plot_var;\n
        dom = "t", xtext = "Time", comp = real, imax = nothing, ylabel = "Re[A(t)]",
        legend=:bottomleft, margins=5Plots.mm)

Plot to compare the pulse profile obtained from numerical simulation and analytical solution
arr_x can be an array of time or frequency points
comp can be real, imag, or abs
"""
function plot_simulton(dopa_sml::NamedTuple; 
        size = (900,350), savefile= true, filename = "./figures/simulton_plot.png",
        xlabel = L"Time t", ylabels = [L"Re[A(t)]", L"Im[A(t)]"], color = nothing,
        colors = palette(:tab10), legend=:bottomleft, margins=5Plots.mm)
    sml_plot_re = plot(frame=:box)
    sml_plot_im = plot(frame=:box)
    
    
    for i in collect(1:2)
        if color == nothing
            color = colors[i]
        end
        
        scatter!(sml_plot_re, dopa_sml.sorted_time, real.(dopa_sml.out_sml[:,i]), label = "Split-step " * L"A_{%$i}(t)", 
            color = color,
            xlabel = xlabel, ylabel = ylabels[1], 
            legend=legend, left_margin=margins, bottom_margin=margins)
        
        plot!(sml_plot_re, dopa_sml.sorted_time, real.(dopa_sml.sol_sml[:,i]), label = "Analytic " * L"A_{%$i}(t)", color = color)
        
        scatter!(sml_plot_im, dopa_sml.sorted_time, imag.(dopa_sml.out_sml[:,i]), label = "Split-step " * L"A_{%$i}(t)", 
            color = color,
            xlabel = xlabel, ylabel = ylabels[2], 
            legend=legend, left_margin=margins, bottom_margin=margins)
        
        plot!(sml_plot_im, dopa_sml.sorted_time, imag.(dopa_sml.sol_sml[:,i]), label = "Analytic " * L"A_{%$i}(t)", 
            color = color)
    end
    
    sml_plot = plot(sml_plot_re, sml_plot_im, layout=(1,2), size=size)
    if savefile
        savefig(filename)
    end
    
    sml_plot
    return sml_plot
end;

end