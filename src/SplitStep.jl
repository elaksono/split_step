"""
SplitStep Module\n
Updated: 6 Mar 2025\n

This module implements the split-step methods for nonlinear pulse propagation simulations.
It provides routines for applying forward dispersion steps and for performing nonlinear updates
using both naive and Runge-Kutta 4th order (RK4) schemes. The module is designed to work with the
nl_class structure defined in NLOclass, which encapsulates the simulation state (field amplitudes,
material properties, dispersion factors, etc.).\n

Key Functions:\n
  - fw_disp!:
      Performs the forward dispersion step by applying an FFT to the current amplitude, multiplying
      by the precomputed dispersion factor (exp_beta_dz), and then applying an inverse FFT.
      Optionally, an FFT plan (p_fft) can be provided for improved performance.
      
  - naive_step_nl!:
      Implements a naive split-step update for the nonlinear part of the propagation.
      It computes the nonlinear contribution (using a supplied nonlinear operator op_nl) and adds it to
      the current field amplitude, then applies the forward dispersion step.
      
  - rk4_step_nl!:
      Provides a Runge-Kutta 4th order update for the nonlinear propagation.
      This function computes intermediate slopes (k1, k2, k3, k4) and combines them to obtain a more
      accurate update of the field amplitude over the propagation step.
      
  - split_step_nl!:
      Orchestrates the overall split-step method over a total propagation distance L.
      It repeatedly applies the chosen nonlinear step method (e.g., naive_step_nl! or rk4_step_nl!)
      along with the forward dispersion step. It can record the field amplitude at each step if needed.

Additional Features:\n
  - Overloaded versions of naive_step_nl! and rk4_step_nl! accept an explicit propagation distance (z)
    to support z-dependent nonlinear operators.
  - The module supports optional FFT planning for both CPU and GPU execution contexts.
  - Linear propagation (i.e., no nonlinear effects) is handled by directly applying the dispersion operator.

This module integrates seamlessly with other components such as Materials, NLO, and NLOclass,
and is essential for simulating the evolution of optical pulses in nonlinear media using the split-step method.
"""
module SplitStep
export fw_disp!, naive_step_nl!, split_step_nl!, rk4_step_nl!

using Revise
using FFTW, Interpolations, LinearAlgebra, Random
using Materials
using NLO, NLOclass

"""
Forward dispersion step

Optional:
p_fft is a tuple containing planned FFT & iFFT
    This should be in-place FFT (planned_fft! and planned_ifft!)
"""
function fw_disp!(dpulse::nl_class; p_fft = nothing, fft_dim = 1, pow = nothing)
    isnothing(p_fft) ? fft!(dpulse.current_amp, fft_dim) : p_fft[1] * dpulse.current_amp
    if isnothing(pow)
        dpulse.current_amp .= dpulse.exp_beta_dz .* dpulse.current_amp
    else
        dpulse.current_amp .= (dpulse.exp_beta_dz).^pow .* dpulse.current_amp
    end
    isnothing(p_fft) ? ifft!(dpulse.current_amp, fft_dim) : p_fft[2] * dpulse.current_amp
    return
end;

"""
naive_step_chi2
Naive split step
Propagated at a finite step of size dz

Inputs:

Optional inputs:

Return out_amp
"""
function naive_step_nl!(dpulse::nl_class, op_nl; 
                        pow = nothing, is_linear = false, p_fft = nothing, cache_id = 1, fft_dim = 1)
    temp = @view dpulse.temp_amp[:,:,cache_id]
    is_linear ? temp = zeros(size(temp)...) : op_nl(dpulse; cache_id = cache_id)
    dpulse.current_amp .= temp .+ dpulse.current_amp
    fw_disp!(dpulse; p_fft = p_fft, fft_dim = fft_dim, pow = pow)
    return
end;

function rk4_step_nl!(dpulse::nl_class, op_nl;
                     is_linear = false, p_fft = nothing) 
    k1 = @view dpulse.temp_amp[:,:,1]
    k2 = @view dpulse.temp_amp[:,:,2]
    k3 = @view dpulse.temp_amp[:,:,3]
    k4 = @view dpulse.temp_amp[:,:,4]
    
    # Remember the starting amp
    cache = @view dpulse.temp_amp[:,:,5]
    
    # Cache = A(t0)
    cache .= dpulse.current_amp
    
    # Calculating ki's
    op_nl(dpulse; cache_id = 1)
    dpulse.current_amp .= k1
    naive_step_nl!(dpulse, op_nl; pow = 0.5,
                  is_linear = true, p_fft = p_fft, cache_id = 1)
    k1 .= dpulse.current_amp  # k1 should be 12(b)

    # Overhead A_IP
    dpulse.current_amp .= cache
    naive_step_nl!(dpulse, op_nl; pow = 0.5,
                  is_linear = true, p_fft = p_fft, cache_id = 5)
    cache .= dpulse.current_amp # AI in cache

    # CHECK THE FOLLOWING 2 LINES
    # dpulse.current_amp .= cache .+ k1  # ?
    # naive_step_nl!(dz/2, dpulse, op_nl;
    #               is_linear = true, p_fft = p_fft, cache_id = 5)  # ?
    
    dpulse.current_amp .= cache .+ (k1 ./ 2)   # AI + k1/2 in current_amp
    op_nl(dpulse; cache_id = 2)
    
    dpulse.current_amp .= cache .+ (k2 ./ 2)   # AI + k2/2 in current_amp
    # k3 = N(dz, AI + k2/2)
    op_nl(dpulse; cache_id = 3)

    dpulse.current_amp .= cache .+ k3   # AI + k3 in current_amp
    naive_step_nl!(dpulse, op_nl; pow = 0.5,
                  is_linear = true, p_fft = p_fft, cache_id = 4)
    op_nl(dpulse; cache_id = 4)  # k4

    # Summing up the contribution
    dpulse.current_amp .= cache .+ (k1 .+ 2 .* k2 .+ 2 .* k3) ./ 6
    naive_step_nl!(dpulse, op_nl; pow = 0.5,
                  is_linear = true, p_fft = p_fft, cache_id = 5)
    dpulse.current_amp .= dpulse.current_amp .+ (k4 ./ 6)
    
    return
end;

# TO BE EDITED...
function naive_step_nl!(z, dpulse::nl_class, op_nl; 
                        pow = nothing, is_linear = false, p_fft = nothing, cache_id = 1, fft_dim = 1)
    temp = @view dpulse.temp_amp[:,:,cache_id]
    is_linear ? temp = zeros(size(temp)...) : op_nl(z, dpulse; cache_id = cache_id, 
        func = dpulse.nl_func, phi_z = dpulse.phi_z)
    dpulse.current_amp .= temp .+ dpulse.current_amp
    fw_disp!(dpulse, p_fft = p_fft, fft_dim = fft_dim)
    return
end;

function rk4_step_nl!(z, dpulse::nl_class, op_nl;
                     is_linear = false, p_fft = nothing) 
    dz = dpulse.dz
    k1 = @view dpulse.temp_amp[:,:,1]
    k2 = @view dpulse.temp_amp[:,:,2]
    k3 = @view dpulse.temp_amp[:,:,3]
    k4 = @view dpulse.temp_amp[:,:,4]
    
    # Remember the starting amp
    cache = @view dpulse.temp_amp[:,:,5]
    
    # Cache = A(t0)
    cache .= dpulse.current_amp
    
    # NL function & phase
    func = dpulse.nl_func
    phi_z = dpulse.phi_z
    
    # Calculating ki's
    op_nl(z, dpulse; cache_id = 1, func = func, phi_z = phi_z)
    dpulse.current_amp .= k1
    naive_step_nl!(dpulse, op_nl; pow = 0.5,
                  is_linear = true, p_fft = p_fft, cache_id = 1)
    k1 .= dpulse.current_amp  # k1 should be 12(b)

    # Overhead A_IP
    dpulse.current_amp .= cache
    naive_step_nl!(dpulse, op_nl; pow = 0.5,
                  is_linear = true, p_fft = p_fft, cache_id = 5)
    cache .= dpulse.current_amp # AI in cache
    
    dpulse.current_amp .= cache .+ (k1 ./ 2)   # AI + k1/2 in current_amp
    op_nl(z + dz/2, dpulse; cache_id = 2, func = func, phi_z = phi_z)
    
    dpulse.current_amp .= cache .+ (k2 ./ 2)   # AI + k2/2 in current_amp
    # k3 = N(dz, AI + k2/2)
    op_nl(z + dz/2, dpulse; cache_id = 3, func = func, phi_z = phi_z)

    dpulse.current_amp .= cache .+ k3   # AI + k3 in current_amp
    naive_step_nl!(dpulse, op_nl; pow = 0.5,
                  is_linear = true, p_fft = p_fft, cache_id = 4)
    op_nl(z + dz, dpulse; cache_id = 4, func = func, phi_z = phi_z)  # k4

    # Summing up the contribution
    dpulse.current_amp .= cache .+ (k1 .+ 2 .* k2 .+ 2 .* k3) ./ 6
    naive_step_nl!(dpulse, op_nl; pow = 0.5,
                  is_linear = true, p_fft = p_fft, cache_id = 5)
    dpulse.current_amp .= dpulse.current_amp .+ (k4 ./ 6)
    
    return
end;

"""
split_step_nl(L, dpulse, op_nl; dz = 0.1, op_nl = op_nl_chi2, 
                    temp_amp = nothing, out_amp = nothing, step_method = naive_step_nl,
                    record_all = false)

Inputs:\n

Optional inputs:\n

Return out_amp
"""
function split_step_nl!(L, dpulse::nl_class, op_nl; step_method = rk4_step_nl!, with_z = false,
                    p_fft = nothing, record_all = false)
    dz = dpulse.dz
    arr_zpos = collect(dz:dz:L)
    amp_type = typeof(dpulse.init_amp[1,1])
    dpulse.current_amp = copy(dpulse.init_amp)
    dpulse.out_amp = record_all ? zeros(amp_type, size(dpulse.init_amp)..., length(arr_zpos)) : zeros(amp_type, size(dpulse.init_amp)...)
    
    if dpulse.mat.prop.linear
        fw_disp!(dpulse; p_fft = p_fft, pow = L/dz)
    else
        for i in 1:length(arr_zpos)
            if with_z
                step_method(arr_zpos[i], dpulse, op_nl, p_fft = p_fft)
            else
                step_method(dpulse, op_nl, p_fft = p_fft)
            end

            if (record_all) dpulse.out_amp[:,:,i] .= dpulse.current_amp end
        end
    end

    if (!record_all) dpulse.out_amp .= dpulse.current_amp end
    return
end;

end
