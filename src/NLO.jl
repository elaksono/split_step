"""
NLO Module: Nonlinear Optical Operators for χ² Processes\n
Updated: 6 Mar 2025

This module implements a set of nonlinear operators for simulating χ²-based (second-order nonlinear)
optical processes. The functions provided here are used to update the field amplitudes during the 
propagation of optical pulses in nonlinear media, and they cover different cases including:
  
1. Degenerate Optical Parametric Amplification (DOPA):
   - op_nl_dopa!(): Applies the nonlinear update for the degenerate optical parametric amplification process.
     In this process, there are two fields involved (Signal and Pump). The operator calculates the changes
     in these fields based on the nonlinear coupling coefficient (κ_matrix) and the specified propagation step (dz).
     
   - The module also provides a CUDA-compatible variant of op_nl_dopa!() that performs the same operation 
     on data arrays residing on the GPU.

2. Nondegenerate Optical Parametric Amplification (χ²):
   - op_nl_chi2!(): Updates the fields for a nondegenerate optical parametric amplification process involving
     three fields (Signal, Idler, and Pump). The nonlinear operator computes the interaction among these fields
     using the coupling matrix.
     
   - A CUDA variant is also available to perform these operations on GPU arrays.

3. Adiabatic Nondegenerate Optical Parametric Amplification:
   - op_nl_adb_chi2!(): Implements the nonlinear operator for the adiabatic version of nondegenerate χ² processes.
     This operator incorporates additional functions (e.g., a modulation function such as cos or another user-defined
     function) and a phase evolution (phi_z) into the nonlinear update. This allows for a more flexible treatment
     of adiabatic processes.
     
   - Similarly, a CUDA variant is provided for adiabatic χ² operators.

Each operator function uses in-place operations (denoted by the "!" in the function names) to update the 
temporary amplitude arrays based on the current amplitudes and the nonlinear interaction coefficients.

Usage:
  - These functions are typically called within a simulation loop that advances the optical fields along the 
    propagation direction. The update rules take into account the propagation step size (dz), the nonlinear 
    coupling coefficients (κ_matrix or κ_matrix_p for GPU), and various optional scaling factors (fac) to control 
    the strength of the interaction.
    
Dependencies:
  - NLOclass: Provides the definition of the nl_class and nl_pulse_p structures that hold the simulation state.
  - CUDA: Enables GPU-accelerated operations for applicable functions.

The module is designed to seamlessly integrate with both CPU and GPU simulations for nonlinear optical pulse 
propagation in χ² media.
"""
module NLO

using NLOclass, CUDA
export op_nl_dopa!, op_nl_chi2!, op_nl_adb_chi2!


"""
op_nl_dopa!()

Nonlinear operators based on chi2-system\n
Degenerate optical parametric amplification process\n
There are two fields (1: Signal, 2: Pump)
"""
function op_nl_dopa!(dpulse::nl_class; fac = (1.0,0.5), cache_id = 1)
    @views signal = dpulse.current_amp[:,1]
    @views pump = dpulse.current_amp[:,2]
    
    @views opN1 = dpulse.temp_amp[:,1,cache_id]
    @views opN2 = dpulse.temp_amp[:,2,cache_id]

    opN1 .=  (fac[1] * dpulse.dz) .* dpulse.κ_matrix .* conj.(signal) .* pump
    opN2 .= -(fac[2] * dpulse.dz) .* dpulse.κ_matrix .* signal .* signal
    return
end;

# TO DO - MODIFY THIS...
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
function op_nl_chi2!(dpulse::nl_class; fac = (1.0,1.0,1.0), cache_id = 1)
    @views signal = dpulse.current_amp[:,1]
    @views idler = dpulse.current_amp[:,2]
    @views pump = dpulse.current_amp[:,3]
    
    @views opN1 = dpulse.temp_amp[:,1,cache_id]
    @views opN2 = dpulse.temp_amp[:,2,cache_id]
    @views opN3 = dpulse.temp_amp[:,3,cache_id]

    # Consider using tuple for this
    opN1 .=  (fac[1] * dpulse.dz) .* dpulse.κ_matrix .* conj.(idler) .* pump
    opN2 .=  (fac[2] * dpulse.dz) .* dpulse.κ_matrix .* conj.(signal) .* pump
    opN3 .= -(fac[3] * dpulse.dz) .* dpulse.κ_matrix .* signal .* idler
    return
end;


# IN-PROGRESS
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

"""
# IN PROGRESS
op_nl_adb_chi2!()\n

Nonlinear operators based on adiabatic chi2-system\n
Non-degenerate adiabatic optical parametric amplification process\n
"""
function op_nl_adb_chi2!(z, dpulse::nl_class; fac = (1.0,1.0,1.0), cache_id = 1, func = cos, phi_z = (z -> 0), g_func = (z -> 1.0))
    @views signal = dpulse.current_amp[:,1]
    @views idler = dpulse.current_amp[:,2]
    @views pump = dpulse.current_amp[:,3]
    
    @views opN1 = dpulse.temp_amp[:,1,cache_id]
    @views opN2 = dpulse.temp_amp[:,2,cache_id]
    @views opN3 = dpulse.temp_amp[:,3,cache_id]

    # Consider using tuple for this
    opN1 .=  (fac[1] * dpulse.dz * func(phi_z(z)) * g_func(z)) .* dpulse.κ_matrix .* conj.(idler) .* pump
    opN2 .=  (fac[2] * dpulse.dz * func(phi_z(z)) * g_func(z)) .* dpulse.κ_matrix .* conj.(signal) .* pump
    opN3 .= -(fac[3] * dpulse.dz * conj(func(phi_z(z))) * g_func(z)) .* dpulse.κ_matrix .* signal .* idler
    return
end;

"""
# IN PROGRESS
op_nl_adb_chi2!()

Nonlinear operators based on chi2-system\n
Non-degenerate adiabatic optical parametric amplification process\n
"""
function op_nl_adb_chi2!(dz, z, dpulse_p::nl_pulse_p; fac = (1.0,1.0,1.0), cache_id = 1, func = cos)
    @views signal = dpulse_p.current_amp_p[:,:,1]
    @views idler = dpulse_p.current_amp_p[:,:,2]
    @views pump = dpulse_p.current_amp_p[:,:,3]
    
    @views opN1 = dpulse_p.temp_amp_p[:,:,1,cache_id]
    @views opN2 = dpulse_p.temp_amp_p[:,:,2,cache_id]
    @views opN3 = dpulse_p.temp_amp_p[:,:,3,cache_id]

    # Consider using tuple for this
    opN1 .=  (fac[1] * dz * func(z)) .* dpulse_p.κ_matrix_p .* conj.(idler) .* pump
    opN2 .=  (fac[2] * dz * func(z)) .* dpulse_p.κ_matrix_p .* conj.(signal) .* pump
    opN3 .= -(fac[3] * dz * func(z)) .* dpulse_p.κ_matrix_p .* signal .* idler
    return
end;

end