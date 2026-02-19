module Basic
using Constant

export freq_lbd_ctr
"""
f_omg_lbd_ctr
Calculating central frequency / wavelength 
"""
function freq_lbd_ctr(;lbd_p = nothing, lbd_s = nothing, lbd_i = nothing)
    if !isnothing(lbd_p) && !isnothing(lbd_s) && isnothing(lbd_i)
        lbd_i = (lbd_p * lbd_s) / (lbd_s - lbd_p)
        if lbd_i < 0
            throw(DomainError(lbd_p, "lbd_p should be shorter than lbd_s"))
        end
    elseif !isnothing(lbd_p) && !isnothing(lbd_i) && isnothing(lbd_s)
        lbd_s = (lbd_p * lbd_i) / (lbd_i - lbd_p)
        if lbd_s < 0
            throw(DomainError(lbd_p, "lbd_p should be shorter than lbd_s"))
        end
    elseif !isnothing(lbd_s) && !isnothing(lbd_i) && isnothing(lbd_p)
        lbd_p = (lbd_s * lbd_i) / (lbd_s - lbd_i)
    else
        throw(ArgumentError(nothing, "two input wavelengths are needed"))
    end
    
    lbd_ctr = 1e-6 .* [lbd_s; lbd_i; lbd_p]
    kvec_ctr = 2*pi ./ lbd_ctr
    freq_ctr = c0*1e-12 ./ lbd_ctr 
    omg_ctr = 2*pi .* freq_ctr
    
    return Dict("lbd" => lbd_ctr, "kvec" => kvec_ctr, "omg" => omg_ctr, "freq" => freq_ctr)
end

export unwrap_phase
"""
unwrap_phase(phase::Vector{<:Real})
    Unwrapping the phase be a continuous function
"""
function unwrap_phase(phase::Vector{<:Real})
    unwrapped = similar(phase)
    unwrapped[1] = phase[1]
    offset = 0.0
    for i in 2:length(phase)
        delta = phase[i] - phase[i-1]
        # Adjust delta if a jump greater than π is detected
        if delta > π
            offset -= 2π
        elseif delta < -π
            offset += 2π
        end
        unwrapped[i] = phase[i] + offset
    end
    return unwrapped
end
    
end