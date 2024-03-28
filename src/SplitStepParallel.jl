module SplitStepParallel
export fw_disp_p!, naive_step_p!, rk4_step_p!, split_step_p!

using Revise
using CUDA
using FFTW, Interpolations, LinearAlgebra, Random
using NLO, NLOclass

function split_step_p!(L, dpulse_p::nl_pulse_p, op_nl;
                       dz = 0.1, step_method = naive_step_p!,
                       p_fft = nothing, fft_dim = 1)
    arr_zpos = collect(dz:dz:L)
    dpulse_p.current_amp_p .= dpulse_p.init_amp_p
    for i in 1:length(arr_zpos)
        step_method(dz, dpulse_p, op_nl; p_fft = p_fft)
    end
    dpulse_p.out_amp_p .= dpulse_p.current_amp_p;
    return
end

function fw_disp_p!(dz, dpulse_p::nl_pulse_p; p_fft = nothing, fft_dim = 1)
    dpulse_p.current_amp_p .= dpulse_p.exp_ω0T_p .* dpulse_p.current_amp_p
    isnothing(p_fft) ? fft!(dpulse_p.current_amp_p, fft_dim) : p_fft[1] * dpulse_p.current_amp_p
    
    dpulse_p.current_amp_p .= exp.((1im * dz) .* dpulse_p.beta_ω_p) .* dpulse_p.current_amp_p
    isnothing(p_fft) ? ifft!(dpulse_p.current_amp_p, fft_dim) : p_fft[2] * dpulse_p.current_amp_p

    dpulse_p.current_amp_p .= dpulse_p.current_amp_p .* conj.(dpulse_p.exp_ω0T_p)
    return
end

function naive_step_p!(dz, dpulse_p::nl_pulse_p, op_nl; 
                       is_linear = false, cache_id = 1, p_fft = nothing, fft_dim = 1)
    @views temp = dpulse_p.temp_amp_p[:,:,:,cache_id]
    is_linear ? temp = CUDA.zeros(size(temp)...) : op_nl(dz, dpulse_p; cache_id = cache_id)
    dpulse_p.current_amp_p .= temp .+ dpulse_p.current_amp_p
    fw_disp_p!(dz, dpulse_p, p_fft = p_fft, fft_dim = fft_dim)
    return
end

function rk4_step_p!(dz, dpulse_p::nl_pulse_p, op_nl;
                     is_linear = false, p_fft = nothing)
    k1 = @view dpulse_p.temp_amp_p[:,:,:,1]
    k2 = @view dpulse_p.temp_amp_p[:,:,:,2]
    k3 = @view dpulse_p.temp_amp_p[:,:,:,3]
    k4 = @view dpulse_p.temp_amp_p[:,:,:,4]
    cache = @view dpulse_p.temp_amp_p[:,:,:,5]
    
    # Cache = A(t0)
    cache .= dpulse_p.current_amp_p
    
    # Calculating ki's
    op_nl(dz, dpulse_p; cache_id = 1)
    dpulse_p.current_amp_p .= k1
    naive_step_p!(dz/2, dpulse_p, op_nl,
                  is_linear = true, p_fft = p_fft, cache_id = 1)
    k1 .= dpulse_p.current_amp_p # k1 should be 12(b)

    # Overhead A_IP
    dpulse_p.current_amp_p .= cache
    naive_step_p!(dz/2, dpulse_p, op_nl;
                  is_linear = true, p_fft = p_fft, cache_id = 5)
    cache .= dpulse_p.current_amp_p # AI in cache
    dpulse_p.current_amp_p .= cache .+ k1
    naive_step_p!(dz/2, dpulse_p, op_nl;
                  is_linear = true, p_fft = p_fft, cache_id = 5)
    
    dpulse_p.current_amp_p .= cache .+ (k1 ./ 2)   # AI + k1/2 in current_amp
    op_nl(dz, dpulse_p; cache_id = 2)
    
    dpulse_p.current_amp_p  .= cache .+ (k2 ./ 2)   # AI + k2/2 in current_amp
    # k3 = N(dz, AI + k2/2)
    op_nl(dz, dpulse_p; cache_id = 3)

    dpulse_p.current_amp_p  .= cache .+ k3   # AI + k3 in current_amp
    naive_step_p!(dz/2, dpulse_p, op_nl;
                  is_linear = true, p_fft = p_fft, cache_id = 4)
    op_nl(dz, dpulse_p; cache_id = 4)  # k4

    # Summing up the contribution
    dpulse_p.current_amp_p .= cache .+ (k1 .+ 2 .* k2 .+ 2 .* k3) ./ 6
    naive_step_p!(dz/2, dpulse_p, op_nl;
                  is_linear = true, p_fft = p_fft, cache_id = 5)
    dpulse_p.current_amp_p .= dpulse_p.current_amp_p  .+ (k4 ./ 6)
    
    return
end

end