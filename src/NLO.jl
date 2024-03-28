# Nonlinear processes
module NLO

using NLOclass, CUDA
export op_nl_dopa!, op_nl_chi2!
"""
op_nl_dopa!()

Nonlinear operators based on chi2-system\n
Degenerate optical parametric amplification process\n
There are two fields (1: Signal, 2: Pump)
"""
function op_nl_dopa!(dz, dpulse::nl_pulse; fac = (1.0,0.5), cache_id = 1)
    @views signal = dpulse.current_amp[:,1]
    @views pump = dpulse.current_amp[:,2]
    
    @views opN1 = dpulse.temp_amp[:,1,cache_id]
    @views opN2 = dpulse.temp_amp[:,2,cache_id]

    opN1 .=  (fac[1] * dz) .* dpulse.κ_matrix .* conj.(signal) .* pump
    opN2 .= -(fac[2] * dz) .* dpulse.κ_matrix .* signal .* signal
    return
end;

"""
op_nl_dopa!()
Operation with CUDA

Nonlinear operators based on chi2-system\n
Degenerate optical parametric amplification process\n
There are two fields (1: Signal, 2: Pump)
"""
function op_nl_dopa!(dz, dpulse_p::nl_pulse_p; fac = (1.0,0.5), cache_id = 1)
    @views signal = dpulse_p.current_amp_p[:,:,1]
    @views pump = dpulse_p.current_amp_p[:,:,2]
    
    @views opN1 = dpulse_p.temp_amp_p[:,:,1,cache_id]
    @views opN2 = dpulse_p.temp_amp_p[:,:,2,cache_id]

    opN1 .=  (fac[1] * dz) .* dpulse_p.κ_matrix_p .* conj.(signal) .* pump
    opN2 .= -(fac[2] * dz) .* dpulse_p.κ_matrix_p .* signal .* signal
    return
end;

"""
op_nl_chi2!()

Nonlinear operators based on chi2-system\n
Non-degenerate optical parametric amplification process\n
"""
function op_nl_chi2!(dz, dpulse::nl_pulse; fac = (1.0,1.0,1.0), cache_id = 1)
    @views signal = dpulse.current_amp[:,1]
    @views idler = dpulse.current_amp[:,2]
    @views pump = dpulse.current_amp[:,3]
    
    @views opN1 = dpulse.temp_amp[:,1,cache_id]
    @views opN2 = dpulse.temp_amp[:,2,cache_id]
    @views opN3 = dpulse.temp_amp[:,3,cache_id]

    # Consider using tuple for this
    opN1 .=  (fac[1] * dz) .* dpulse.κ_matrix .* conj.(idler) .* pump
    opN2 .=  (fac[2] * dz) .* dpulse.κ_matrix .* conj.(signal) .* pump
    opN3 .= -(fac[3] * dz) .* dpulse.κ_matrix .* signal .* idler
    return
end;

function op_nl_chi2!(dz, dpulse_p::nl_pulse_p; fac = (1.0,1.0,1.0), cache_id = 1)
    @views signal = dpulse_p.current_amp_p[:,:,1]
    @views idler = dpulse_p.current_amp_p[:,:,2]
    @views pump = dpulse_p.current_amp_p[:,:,3]
    
    @views opN1 = dpulse_p.temp_amp_p[:,:,1,cache_id]
    @views opN2 = dpulse_p.temp_amp_p[:,:,2,cache_id]
    @views opN3 = dpulse_p.temp_amp_p[:,:,3,cache_id]

    # Consider using tuple for this
    opN1 .=  (fac[1] * dz) .* dpulse_p.κ_matrix_p .* conj.(idler) .* pump
    opN2 .=  (fac[2] * dz) .* dpulse_p.κ_matrix_p .* conj.(signal) .* pump
    opN3 .= -(fac[3] * dz) .* dpulse_p.κ_matrix_p .* signal .* idler
    return
end;

end