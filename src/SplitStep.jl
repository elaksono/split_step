module SplitStep
export fw_disp!, naive_step_nl!, split_step_nl!, rk4_step_nl!

using Revise
using FFTW, Interpolations, LinearAlgebra, Random
using NLO, NLOclass

"""
Forward dispersion step

Optional:
p_fft is a tuple containing planned FFT & iFFT
    This should be in-place FFT (planned_fft! and planned_ifft!)
"""
function fw_disp!(dz, dpulse; p_fft = nothing, fft_dim = 1)
    dpulse.current_amp .= dpulse.exp_ω0T .* dpulse.current_amp
    isnothing(p_fft) ? fft!(dpulse.current_amp, fft_dim) : p_fft[1] * dpulse.current_amp
    
    dpulse.current_amp .= exp.((1im * dz) .* dpulse.beta_ω) .* dpulse.current_amp
    isnothing(p_fft) ? ifft!(dpulse.current_amp, fft_dim) : p_fft[2] * dpulse.current_amp

    dpulse.current_amp .= dpulse.current_amp .* conj.(dpulse.exp_ω0T)
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
function naive_step_nl!(dz, dpulse, op_nl; 
                        is_linear = false, p_fft = nothing, cache_id = 1, fft_dim = 1)
    temp = @view dpulse.temp_amp[:,:,cache_id]
    is_linear ? temp = zeros(size(temp)...) : op_nl(dz, dpulse; cache_id = cache_id)
    dpulse.current_amp .= temp .+ dpulse.current_amp
    fw_disp!(dz, dpulse, p_fft = p_fft, fft_dim = fft_dim)
    return
end;

function rk4_step_nl!(dz, dpulse, op_nl;
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
    op_nl(dz, dpulse; cache_id = 1)
    dpulse.current_amp .= k1
    naive_step_nl!(dz/2, dpulse, op_nl,
                  is_linear = true, p_fft = p_fft, cache_id = 1)
    k1 .= dpulse.current_amp  # k1 should be 12(b)

    # Overhead A_IP
    dpulse.current_amp .= cache
    naive_step_nl!(dz/2, dpulse, op_nl;
                  is_linear = true, p_fft = p_fft, cache_id = 5)
    cache .= dpulse.current_amp # AI in cache
    dpulse.current_amp .= cache .+ k1
    naive_step_nl!(dz/2, dpulse, op_nl;
                  is_linear = true, p_fft = p_fft, cache_id = 5)
    
    dpulse.current_amp .= cache .+ (k1 ./ 2)   # AI + k1/2 in current_amp
    op_nl(dz, dpulse; cache_id = 2)
    
    dpulse.current_amp .= cache .+ (k2 ./ 2)   # AI + k2/2 in current_amp
    # k3 = N(dz, AI + k2/2)
    op_nl(dz, dpulse; cache_id = 3)

    dpulse.current_amp .= cache .+ k3   # AI + k3 in current_amp
    naive_step_nl!(dz/2, dpulse, op_nl;
                  is_linear = true, p_fft = p_fft, cache_id = 4)
    op_nl(dz, dpulse; cache_id = 4)  # k4

    # Summing up the contribution
    dpulse.current_amp .= cache .+ (k1 .+ 2 .* k2 .+ 2 .* k3) ./ 6
    naive_step_nl!(dz/2, dpulse, op_nl;
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
function split_step_nl!(L, dpulse, op_nl; dz = 0.1, step_method = rk4_step_nl!, 
                    p_fft = nothing, record_all = false)
    arr_zpos = collect(dz:dz:L)
    amp_type = typeof(dpulse.init_amp[1,1])
    dpulse.current_amp = copy(dpulse.init_amp)
    dpulse.out_amp = record_all ? zeros(amp_type, size(dpulse.init_amp)..., length(arr_zpos)) : zeros(amp_type, size(dpulse.init_amp)...)
    
    for i in 1:length(arr_zpos)
        step_method(dz, dpulse, op_nl, p_fft = p_fft)
        if (record_all) dpulse.out_amp[:,:,i] .= dpulse.current_amp end
    end

    if (!record_all) dpulse.out_amp .= dpulse.current_amp end
    return
end;

end
