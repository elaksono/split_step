"""
ECPOclass Module\n
Updated: 6 Mar 2025

This module defines the core classes and functions for creating, configuring, and managing ECPO
(optical pulse propagation) objects used in nonlinear optical simulations. It provides tools to
initialize ECPO instances from JSON parameter files, set up the simulation environment, and manage
propagation routines, including both standard and adiabatic poling variants. Additionally, the module
supports parallel implementations for resonant ECPO operations with GPU acceleration using CUDA.

Key Features:
  • Input Parsing and Initialization:
      - Reads simulation parameters from JSON files.
      - Creates output directories and stores input parameters for reproducibility.
  • ECPO Object Types:
      - `ecpo`: Standard ECPO object containing parameters for pump, signal, and idler pulses.
      - `ecpo_adb`: ECPO variant with adiabatic poling/tapering for optimized phase matching.
      - `ecpo_p`: A parallel ECPO class for resonant operations, handling multiple ECPO instances on GPUs.
  • Simulation Setup:
      - Computes key physical quantities (wavelengths, angular frequencies, pulse energies).
      - Initializes waveguide and dispersive element properties.
      - Configures modulation functions for phase and amplitude.
      - Sets up split-step propagation structures for nonlinear pulse evolution.
  • Utility Functions:
      - Provides restart functionality and unified type alias (`ecpo_class`) for ease of use.

Dependencies:
  - Standard libraries: Dates, JSON, CSV, DataFrames.
  - GPU support: CUDA.
  - Custom modules: NLO, NLOclass, SplitStep, SplitStepParallel, Pulse, Modulation, Materials.

Usage Example:
    using ECPOclass
    my_ecpo = ecpo("path/to/parameters.json")
    parallel_ecpo = ecpo_p([my_ecpo], "simulation_label")

This module is intended for researchers and engineers in the field of nonlinear optics, offering a robust
framework for simulating advanced optical pulse propagation phenomena.
"""
module ECPOclass

using CUDA
using Dates
using JSON, CSV, DataFrames
using NLO, NLOclass, SplitStep, SplitStepParallel
using Pulse, Modulation
using Materials

export ecpo, ecpo_adb, ecpo_p, restart_ecpo
export ecpo_class

"""
ECPO object

ecpo(input_string::String; with_T_cav = false)
    input_string
"""
mutable struct ecpo
    # IO Data
    input_data::String
    save_folder::String
    save_label::String
    
    # Pump, signal and idler
    lambda_vec::Vector{Real}
    omega_vec::Vector{Real}
    
    # Number of rounds
    n_round::Integer
    n_saves::Integer
    
    # Cavity
    in_tab::Vector{Real}
    out_tab::Vector{Real}
    α_tab::Vector{Real}
    L_cav::Real
    T_cav::Real
    
    # Modulation frequencies
    f_mod::Real
    T_mod::Real
    
    # Modulation functions
    phase_mod_func
    amp_mod_func
    pars_pm
    pars_am
    
    # Waveguide
    L_wg::Real
    wg_mat::Material
    
    # Dispersive element
    L_smf::Real
    smf_mat::Material
    
    # Split-step struct
    pulse_opa::pars_pulse
    ecpo_pulse::nl_pulse
end

function ecpo(input_string::String; with_T_cav = false)
    input_data = input_string
    raw = JSON.parse(open(input_data))
    
    # Create output directory
    current_date = Dates.format(now(), "mm-dd-yy")
    outdir = replace(dirname(input_string), "/input/" => "/output/")
    outdir = outdir * "_" * current_date
    
    if !isdir(outdir)
        mkdir(outdir)
        println("New directory created: $(outdir)")
    end
    
    # Save label
    save_folder = outdir
    save_label = raw["save_label"]
    copypath = outdir * "/parameters.json"
    
    if isfile(copypath)
        rm(copypath)
    end
    cp(input_data, copypath)
    
    # SIGNAL, IDLER, PUMP
    ħ = 1.05457 * 1e-34; c0 = 3e8;
    λs = raw["lambda_s"]
    λp = raw["lambda_p"]
    λi = round(λs * λp / (λs - λp), digits=5);
    
    # In THz and J
    ωs = 2*pi * c0/λs * 1e6 * 1e-12; Es = ħ * ωs * 1e12;
    ωi = 2*pi * c0/λi * 1e6 * 1e-12; Ei = ħ * ωi * 1e12;
    ωp = 2*pi * c0/λp * 1e6 * 1e-12; Ep = ħ * ωp * 1e12;
    
    lambda_vec = [λs; λi; λp]
    omega_vec = [ωs; ωi; ωp]
    
    # NLOclass for split-step
    dz = raw["dz"]
    pulse_ref = raw["pulse_ref"]
    fac = raw["fac"]
    phase_mismatch = raw["phase_mismatch"]
    
    # Rounds
    n_round = raw["n_round"]
    n_saves = raw["n_saves"]
    
    # Cavity
    in_tab = [raw["in_signal"]; raw["in_idler"]; raw["in_pump"]]
    out_tab = [raw["out_signal"]; raw["out_idler"]; raw["out_pump"]]
    α_tab = [raw["loss_signal"]; raw["loss_idler"]; raw["loss_pump"]]
    L_cav = raw["L_cav"]
    T_cav = L_cav / c0 * 1e12
    
    # Modulation freqs
    f_mod = raw["f_mod"] * 1e-3 # THz
    T_mod = 1/f_mod
    
    # Modulation functions
    phase_mod_func = modulation_func[raw["phase_mod_f"]] # Hasn't been defined
    amp_mod_func = modulation_func[raw["amp_mod_f"]]
    pars_pm = [raw["pm_depth"], raw["pm_phase"]]
    pars_am = [raw["am_depth"], raw["am_phase"]]
    
    # Waveguide
    L_wg = raw["L_wg"]
    wg_mat = eval(Meta.parse(raw["wg_mat"]))
    
    if raw["kappa_unit"] == "flux"
        wg_mat.kappa = raw["wg_kappa"]
    elseif raw["kappa_unit"] == "intensity"
        # Unit: /W^0.5 mm => (ps^0.5) /mm
        wg_mat.kappa = raw["wg_kappa"] * sqrt(ħ * ωs * ωi / ωp * 1e12) * 1e6
    end
    
    # Dispersive elements
    smf_mat = smf28
    smf_mat.prop.linear = true
    dcf = raw["dcf"]
    if dcf > 0.0
        L_smf = -2*(1-1/dcf)*(beta_2_func(wg_mat, ωs)/beta_2_func(smf_mat, ωs))*L_wg
    else
        L_smf = raw["L_smf"]
    end
    
    pulse_opa = pars_pulse()
    pulse_opa.amp_max = [raw["signal_max"]; raw["idler_max"]; raw["pump_max"]]
    p_th = raw["p_th"]
    
    if p_th > 0.0
        c_th = atanh(sqrt(α_tab[1])) / (wg_mat.kappa * L_wg)
        pulse_opa.amp_max[3] = p_th * c_th
    end
    
    pulse_opa.omg_ctr = omega_vec
    pulse_opa.T_pulse = [raw["signal_T"]; raw["idler_T"]; raw["pump_T"]]
    pulse_opa.N_div = raw["N_div"]
    
    pulse_opa.f_pulse = [pulse_func[raw["signal_func"]]; 
                         pulse_func[raw["idler_func"]]; 
                         pulse_func[raw["pump_func"]]]
    if (pulse_opa.f_pulse[1] == pulse_opa.f_pulse[2] == pulse_opa.f_pulse[3])
        pulse_opa.f_pulse = pulse_opa.f_pulse[1]
    end
    
    if with_T_cav
        ecpo_pulse = nl_pulse(wg_mat, pulse_opa; dz = dz, pulse_ref = pulse_ref, fac = fac, phase_mismatch = phase_mismatch, 
            Tm = T_cav, calculate_dk = true)
    else
        ecpo_pulse = nl_pulse(wg_mat, pulse_opa; dz = dz, pulse_ref = pulse_ref, fac = fac, phase_mismatch = phase_mismatch, 
            calculate_dk = true)
    end
    
    ecpo(input_data, save_folder, save_label, 
        lambda_vec, omega_vec,
        n_round, n_saves,
        in_tab, out_tab, α_tab, L_cav, T_cav,
        f_mod, T_mod,
        phase_mod_func, amp_mod_func, pars_pm, pars_am,
        L_wg, wg_mat, L_smf, smf_mat,
        pulse_opa, ecpo_pulse)
end

function restart_ecpo(ecpo_var::ecpo)
    return ecpo(ecpo_var.input_data)
end

# ==========================
# OPO with adiabatic poling
mutable struct ecpo_adb
    # IO Data
    input_data::String
    save_folder::String
    save_label::String
    
    # Pump, signal and idler
    lambda_vec::Vector{Real}
    omega_vec::Vector{Real}
    
    # Number of rounds
    n_round::Integer
    n_saves::Integer
    
    # Cavity
    in_tab::Vector{Real}
    out_tab::Vector{Real}
    α_tab::Vector{Real}
    L_cav::Real
    T_cav::Real
    
    # Modulation frequencies
    f_mod::Real
    T_mod::Real
    
    # Modulation functions
    phase_mod_func
    amp_mod_func
    pars_pm
    pars_am
    
    # Poling/ tapering functions
    phi_z_func
    phi_z_pars
    g_func
    g_pars
    
    # Waveguide
    L_wg::Real
    wg_mat::Material
    
    # Dispersive element
    L_smf::Real
    smf_mat::Material
    
    # Split-step struct
    pulse_opa::pars_pulse
    ecpo_pulse::nl_pulse_adb
end

function ecpo_adb(input_string::String; with_T_cav = false)
    input_data = input_string
    raw = JSON.parse(open(input_data))
    
    # Create output directory
    current_date = Dates.format(now(), "mm-dd-yy")
    outdir = replace(dirname(input_string), "/input/" => "/output/")
    outdir = outdir * "_" * current_date
    
    if !isdir(outdir)
        mkdir(outdir)
        println("New directory created: $(outdir)")
    end
    
    # Save label
    save_folder = outdir
    save_label = raw["save_label"]
    copypath = outdir * "/parameters.json"
    
    if isfile(copypath)
        rm(copypath)
    end
    cp(input_data, copypath)
    
    # SIGNAL, IDLER, PUMP
    ħ = 1.05457 * 1e-34; c0 = 3e8;
    λs = raw["lambda_s"]
    λp = raw["lambda_p"]
    λi = round(λs * λp / (λs - λp), digits=5);
    
    # In THz and J
    ωs = 2*pi * c0/λs * 1e6 * 1e-12; Es = ħ * ωs * 1e12;
    ωi = 2*pi * c0/λi * 1e6 * 1e-12; Ei = ħ * ωi * 1e12;
    ωp = 2*pi * c0/λp * 1e6 * 1e-12; Ep = ħ * ωp * 1e12;
    
    lambda_vec = [λs; λi; λp]
    omega_vec = [ωs; ωi; ωp]
    
    # NLOclass for split-step
    dz = raw["dz"]
    pulse_ref = raw["pulse_ref"]
    fac = raw["fac"]
    qpm_lambda = raw["qpm_lambda"]
    
    # Rounds
    n_round = raw["n_round"]
    n_saves = raw["n_saves"]
    
    # Cavity
    in_tab = raw["in_coupling"]
    out_tab = raw["out_coupling"]
    α_tab = raw["loss"]
    L_cav = raw["L_cav"]
    T_cav = L_cav / c0 * 1e12
    
    # Modulation freqs
    f_mod = raw["f_mod"] * 1e-3 # THz
    T_mod = 1/f_mod
    
    # Modulation functions
    phase_mod_func = modulation_func[raw["phase_mod_f"]] # Hasn't been defined
    amp_mod_func = modulation_func[raw["amp_mod_f"]]
    pars_pm = raw["phase_pars"]
    pars_am = raw["amp_pars"]
    
    # Poling/ Tapering functions
    phi_z_func = phase_z_func[raw["phi_z_func"]]
    phi_z_pars = raw["phi_z_pars"]
    g_func = taper_func[raw["g_func"]]
    g_pars = raw["g_pars"]
    g_pars[1] = raw["L_wg"]
    
    # Waveguide
    L_wg = raw["L_wg"]
    wg_mat = eval(Meta.parse(raw["wg_mat"]))
    
    if raw["kappa_unit"] == "flux"
        wg_mat.kappa = raw["wg_kappa"]
    elseif raw["kappa_unit"] == "intensity"
        # Unit: /W^0.5 mm => (ps^0.5) /mm
        wg_mat.kappa = raw["wg_kappa"] * sqrt(ħ * ωs * ωi / ωp * 1e12) * 1e6
    end
    
    # Dispersive elements
    smf_mat = smf28
    smf_mat.prop.linear = true
    dcf = raw["dcf"]
    if dcf > 0.0
        L_smf = -2*(1-1/dcf)*(beta_2_func(wg_mat, ωs)/beta_2_func(smf_mat, ωs))*L_wg
    else
        L_smf = raw["L_smf"]
    end
    
    pulse_opa = pars_pulse()
    pulse_opa.amp_max = raw["amp_max"]
    p_th = raw["p_th"]
    
    if p_th > 0.0
        c_th = atanh(sqrt(α_tab[1])) / (wg_mat.kappa * L_wg)
        
        if raw["poling"] == "adiabatic"
            dk_1 = phi_z_pars[end]
            c_th_2 = (1/wg_mat.kappa) * sqrt(dk_1/ (2*pi) * log(1/(1-α_tab[1])))
            if (c_th_2 > c_th) c_th = c_th_2 end
        end
        
        pulse_opa.amp_max[3] = p_th * c_th
    end
    
    pulse_opa.omg_ctr = omega_vec
    pulse_opa.T_pulse = raw["pulse_T"]
    pulse_opa.N_div = raw["N_div"]
    
    amp_func = raw["amp_func"]
    pulse_opa.f_pulse = [pulse_func[amp_func[i]] for i in 1:length(amp_func)]
    if (pulse_opa.f_pulse[1] == pulse_opa.f_pulse[2] == pulse_opa.f_pulse[3])
        pulse_opa.f_pulse = pulse_opa.f_pulse[1]
    end
    
    if with_T_cav
        ecpo_pulse = nl_pulse_adb(wg_mat, pulse_opa; dz = dz, pulse_ref = pulse_ref, fac = fac, qpm_lambda = qpm_lambda, 
            Tm = T_cav, calculate_dk = true, phi_z_func = phi_z_func, phi_z_pars = phi_z_pars, g_func = g_func, g_pars = g_pars)
    else
        ecpo_pulse = nl_pulse_adb(wg_mat, pulse_opa; dz = dz, pulse_ref = pulse_ref, fac = fac, qpm_lambda = qpm_lambda,  
            calculate_dk = true, phi_z_func = phi_z_func, phi_z_pars = phi_z_pars, g_func = g_func, g_pars = g_pars)
    end
    
    ecpo_adb(input_data, save_folder, save_label, 
        lambda_vec, omega_vec,
        n_round, n_saves,
        in_tab, out_tab, α_tab, L_cav, T_cav,
        f_mod, T_mod,
        phase_mod_func, amp_mod_func, pars_pm, pars_am,
        phi_z_func, phi_z_pars, g_func, g_pars,
        L_wg, wg_mat, L_smf, smf_mat,
        pulse_opa, ecpo_pulse)
end

function restart_ecpo(ecpo_var::ecpo_adb)
    return ecpo(ecpo_var.input_data)
end

ecpo_class = Union{ecpo, ecpo_adb}

"""
Parallel ECPO class
Assumption: resonant operation of the ECPO
"""
mutable struct ecpo_p
    # Data
    ecpo_vec::Vector{ecpo}
    
    # Save
    save_dir::String
    file_label::String
    
    # Number of rounds
    n_round::Integer
    n_saves::Integer
    
    # Waveguide
    L_wg::Real
    
    # Cavity
    in_tot_p::CuArray
    out_tot_p::CuArray
    α_tot_p::CuArray
    
    # Modulation
    pm_p::CuArray
    am_p::CuArray
    
    # Split-step struct
    ecpo_pulse_p::nl_pulse_p
end

function ecpo_p(ecpo_vec::Vector{ecpo}, file_label::String)
    nl_vec = [ecpo_vec[i].ecpo_pulse for i in 1:length(ecpo_vec)]
    arr_time = nl_vec[1].arr_time
    
    # Save
    save_folder = ecpo_vec[1].save_folder
    
    # Number of rounds
    n_round = ecpo_vec[1].n_round
    n_saves = ecpo_vec[1].n_saves
    
    # Waveguide
    L_wg = ecpo_vec[1].L_wg
    
    # Loss arrays (n_bins x n_cases x n_modes)
    in_tot_p = CUDA.zeros(Float32, length(arr_time), length(ecpo_vec), 3)    
    out_tot_p = CUDA.zeros(Float32, length(arr_time), length(ecpo_vec), 3)
    α_tot_p = CUDA.zeros(Float32, length(arr_time), length(ecpo_vec), 3)
    
    for i in 1:length(ecpo_vec)
        in_tot = reshape(ecpo_vec[i].in_tab, (1,:))
        in_tot = repeat(in_tot, length(arr_time))
        in_tot_p[:,i,:] .= CuArray{Float32}(in_tot)
        
        out_tot = reshape(ecpo_vec[i].out_tab, (1,:))
        out_tot = repeat(out_tot, length(arr_time))
        out_tot_p[:,i,:] .= CuArray{Float32}(out_tot)
        
        α_tot = reshape(ecpo_vec[i].α_tab, (1,:))
        α_tot = repeat(α_tot, length(arr_time))
        α_tot_p[:,i,:] .= CuArray{Float32}(α_tot)
    end
    
    # Modulation
    f_phase_mod = ecpo_vec[1].phase_mod_func
    f_amp_mod = ecpo_vec[1].amp_mod_func
    pm_p = CUDA.zeros(ComplexF32, length(arr_time), length(ecpo_vec), 3)
    am_p = CUDA.zeros(Float32, length(arr_time), length(ecpo_vec), 3)
    
    for i in 1:length(ecpo_vec)
        dev = ecpo_vec[i]
        pm_p[:,i,:] .= CuArray(exp.(-1im .* [f_phase_mod(t, dev.T_mod, dev.pars_pm[1], dev.pars_pm[2]) for t in arr_time]))
        am_p[:,i,:] .= CuArray(cos.([f_amp_mod(t, dev.T_mod, dev.pars_am[1], dev.pars_am[2]) for t in arr_time]))
    end
    
    # Split-step struct
    ecpo_pulse_p = nl_pulse_p(nl_vec)
    
    ecpo_p(ecpo_vec, 
        save_folder, file_label,
        n_round, n_saves, L_wg,
        in_tot_p, out_tot_p, α_tot_p,
        pm_p, am_p, 
        ecpo_pulse_p)
end

end