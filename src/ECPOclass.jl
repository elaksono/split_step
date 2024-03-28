module ECPOclass

using CUDA
using JSON, CSV, DataFrames
using NLO, NLOclass, SplitStep, SplitStepParallel
using Pulse, Modulation
export ecpo, ecpo_p, pulse_func, modulation_func

# INTERNAL DICTIONARY
# Pulse initiation
pulse_func = Dict("const" => ff_const, "rand" => ff_rand,
    "gauss" => ff_gauss, "gauss_rand" => ff_gauss_rand, "sech" => ff_sech, "sech1" => ff_sech1);

# Modulation function
# The function form: f(t, T, A, ϕ)
modulation_func = Dict("sin" => fm_sin, "cos" => fm_cos, "sin2" => fm_sin2, "cos2" => fm_cos2)

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
    
    # Waveguide
    L_wg::Real
    
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
    
    # Split-step struct
    ecpo_pulse::nl_pulse
end

function ecpo(input_string::String)
    input_data = input_string
    raw = JSON.parse(open(input_data))
    
    # Create output directory
    outdir = replace(dirname(input_string), "./input/" => "./output/")
    if !isdir(outdir)
        mkdir(outdir)
        println("New directory created: $(outdir)")
    end
    
    # Save label
    save_folder = raw["save_folder"]
    save_label = raw["save_label"]
    
    # SIGNAL, IDLER, PUMP
    ħ = 1.05457 * 1e-34; c0 = 3e8;
    λs = raw["lambda_s"]
    λp = raw["lambda_p"]
    λi = round(λs * λp / (λs - λp), digits=5);
    
    # In THz and J
    ωs = 2*pi * (3*1e8)/λs * 1e6 * 1e-12; Es = ħ * ωs * 1e12;
    ωi = 2*pi * (3*1e8)/λi * 1e6 * 1e-12; Ei = ħ * ωi * 1e12;
    ωp = 2*pi * (3*1e8)/λp * 1e6 * 1e-12; Ep = ħ * ωp * 1e12;
    
    lambda_vec = [λs; λi; λp]
    omega_vec = [ωs; ωi; ωp]
    
    # Rounds
    n_round = raw["n_round"]
    n_saves = raw["n_saves"]
    
    # Waveguide
    L_wg = raw["L_wg"]
    
    # Cavity
    in_tab = [raw["in_signal"]; raw["in_idler"]; raw["in_pump"]]
    out_tab = [raw["out_signal"]; raw["out_idler"]; raw["out_pump"]]
    α_tab = [raw["loss_signal"]; raw["loss_idler"]; raw["loss_pump"]]
    L_cav = raw["L_cav"]
    T_cav = L_cav / c0 * 1e12
    
    # Modulation freqs
    f_mod = raw["f_mod"]
    T_mod = 1/f_mod
    
    # Modulation functions
    phase_mod_func = modulation_func[raw["phase_mod_f"]] # Hasn't been defined
    amp_mod_func = modulation_func[raw["amp_mod_f"]]
    pars_pm = (raw["pm_depth"], raw["pm_phase"])
    pars_am = (raw["am_depth"], raw["am_phase"])
    
    # Pulse
    opa = pars_nl()
    
    file_input = raw["csv_folder"] * raw["disp_csv"]
    opa.βnl = CSV.File(file_input) |> Tables.matrix
    opa.ωnl = [raw["omega_signal"]; raw["omega_idler"]; raw["omega_pump"]]
    opa.κnl = raw["kappa"] #
    
    pulse_opa = pars_pulse()
    pulse_opa.amp_max = [raw["signal_max"]; raw["idler_max"]; raw["pump_max"]]
    pulse_opa.T_pulse = [raw["signal_T"]; raw["idler_T"]; raw["pump_T"]]
    pulse_opa.N_div = raw["N_div"]
    pulse_opa.f_pulse = [pulse_func[raw["signal_func"]]; 
                         pulse_func[raw["idler_func"]]; 
                         pulse_func[raw["pump_func"]]]
    
    ecpo_pulse = nl_pulse(opa, pulse_opa)
    
    ecpo(input_data, save_folder, save_label, 
        lambda_vec, omega_vec,
        n_round, n_saves,
        L_wg,
        in_tab, out_tab, α_tab, L_cav, T_cav,
        f_mod, T_mod,
        phase_mod_func, amp_mod_func, pars_pm, pars_am,
        ecpo_pulse)
end


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