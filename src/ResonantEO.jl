module ResonantEO

using NLO, NLOclass, ECPO, ECPOclass
export Acav, AcavFT

# Analytic solution
function Acav(t, resonant_eo::ecpo; Δ = nothing, Ain = nothing)
    γ = resonant_eo.out_tab[1]
    Γ = resonant_eo.α_tab[1]
    ϕpm = resonant_eo.pars_pm[1]
    Ωpm = 2*pi*resonant_eo.f_mod
    G = (abs(ϕpm)/(Γ))^2
    if isnothing(Δ)
        Tcav = resonant_eo.T_cav
        Ωcav = 2 * pi / Tcav
        dΩtab = (mod(Ωpm, Ωcav), mod(Ωpm, Ωcav) - Ωcav)
        min_id = argmin(abs.(dΩtab))
        Δ = dΩtab[min_id]/(Γ/2)
    end
    
    if isnothing(Ain)
        Ain = resonant_eo.ecpo_pulse.pulse.amp_max[1]
    end
    
    output = sqrt(4γ^2/Γ^2) * Ain/ (1 + 1im*Δ + 2*1im*sqrt(G)*cos(Ωpm*t))
    return output
end;

function AcavFT(k, resonant_eo::ecpo; Δ = nothing, Ain = nothing)
    γ = resonant_eo.out_tab[1]
    Γ = resonant_eo.α_tab[1]
    ϕpm = resonant_eo.pars_pm[1]
    Ωpm = 2*pi*resonant_eo.f_mod
    G = (abs(ϕpm)/(Γ))^2
    
    if isnothing(Δ)
        Tcav = resonant_eo.T_cav
        Ωcav = 2 * pi / Tcav
        dΩtab = (mod(Ωpm, Ωcav), mod(Ωpm, Ωcav) - Ωcav)
        min_id = argmin(abs.(dΩtab))
        Δ = dΩtab[min_id]/(Γ/2)
    end
    
    if isnothing(Ain)
        Ain = resonant_eo.ecpo_pulse.pulse.amp_max[1]
    end
    
    expBeta = (-1 + sqrt(1 + 4*G/(1 + 1im*Δ)^2)) * (1 + 1im*Δ) / (2*sqrt(G))
    A0G = (2γ/Γ) * Ain * expBeta / (sqrt(G)*(1+expBeta^2))
    output = A0G * (-1im*expBeta)^abs(k)
    return output
end;

end