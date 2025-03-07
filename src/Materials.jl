"""
Materials Module: Dispersion & Optical Material Models
Updated: 6 Mar 2025

This module implements a variety of material models and dispersion functions
for optical simulations. 

It includes constant, Taylor-series, Sellmeier (and its variant),
rational, and Laurent dispersion models, as well as utilities for converting full
dispersion data into Taylor series form. Designed for simulating optical waveguides
and nonlinear phenomena, the module handles unit conversions, parameter setting,
and common material definitions.

Units:
omega is in THz
lbd is in micron
beta is in mm^-1
length, width, height in cm
angle in degree
Tc in Celsius

Key function:
n_func(mat::Material, omega::Float64)
"""
module Materials

using Constant
using CUDA, FFTW
using NPZ

export Material, ConstantMaterial, TaylorMaterial, SellmeierMaterial, SellmeierMaterial_2, RationalMaterial, LaurentMaterial
export beta_taylor, beta_sellmeier, beta_rational, beta_laurent
export beta_func, beta_1_func, beta_2_func, beta_3_func, beta_4_func, n_func
export convert_to_TaylorMaterial


"""
Taylor series dispersion
"""
function beta_taylor(omega::Float64, beta_arr)
    return sum(beta_arr .* [omega.^j ./ factorial(j) for j in collect(0:length(beta_arr)-1)])
end;

"""
Fu=ll dispersion in Sellmeier form
"""
function beta_sellmeier(omega::Float64, Avec::Vector{Float64}, Bvec::Vector{Float64})
    k0 = (omega * 1e12/ c0) * 1e-6
    lbd = 2 * pi / k0
    n0 = sqrt(1 + sum(Avec .* (lbd^2) ./ ((lbd^2) .- Bvec)))
    beta = n0 * k0 * 1e3
    return beta
end;

"""
Full dispersion in rational form
"""
function beta_rational(omega::Float64, Avec::Vector{Float64}, Bvec::Vector{Float64})
    k0 = (omega * 1e12/ c0) * 1e-6
    lbd = 2 * pi / k0
    num = sum(Avec .* (lbd .^ collect(0:-2:-2*(length(Avec)-1))))
    den = 1 + sum(Bvec .* (lbd .^ collect(-2:-2:-2*length(Bvec))))
    beta = sqrt(1 + num/ den) * k0 * 1e3
    return beta
end;

"""
Full dispersion in Laurent series
"""
function beta_laurent(omega::Float64, Cvec::Vector{Float64}, Pvec::Vector{Float64})
    k0 = (omega * 1e12/ c0) * 1e-6
    lbd = 2 * pi / k0
    beta = (k0 * 1e3) * sum(Cvec .* (lbd .^ Pvec))
    return beta
end;

Base.@kwdef mutable struct CommonMaterialProperties
    name::String
    ll::Float64
    ww::Float64
    hh::Float64
    tt::Float64
    angle::Float64
    Tc::Float64
    linear::Bool
end

"""
Material:\n
AbstractType for computation on CPU\n

Subtypes:\n
- ConstantMaterial\n
- TaylorMaterial\n
- SellmeierMaterial\n
- LaurentMaterial\n
"""
abstract type Material end

# Non-dispersive Material
"""
ConstantMaterial(name, ll, n0, kappa; ww=0, hh=0, tt=0, angle=0, Tc=30, linear=false)

Args:\n
name (String)\n
ll (Float)\n
n0 (Float): refractive index (flat)\n
kappa (Float): non-linear coefficient (flat)\n

Keyword arguments:\n
ww (Float)\n
    Default = 0 (bulk)\n
hh (Float)\n
    Default = 0 (bulk)\n
Tc (Float)\n
    Default = 30 (room temperature)\n
linear (Bool)\n
    Is it just a dispersive material?\n
    Default = false
"""
mutable struct ConstantMaterial <: Material
    prop::CommonMaterialProperties
    n0::Float64
    kappa::Float64
    
    function ConstantMaterial(name::String, ll::Float64, n0::Float64, kappa::Float64; ww=0.0, hh=0.0, tt=0.0, angle=0.0, Tc=30.0, linear=false)
        new(CommonMaterialProperties(name, ll, ww, hh, tt, angle, Tc, linear), n0, kappa)
    end
end

function beta_func(mat::ConstantMaterial, omega::Float64)
    beta = mat.n0 * (omega * 1e12 / c0) * 1e-3
    return beta
end

# Taylor Material
"""
TaylorMaterial(name, ll, omega_c, n_order, n_waves, beta_array, kappa; ww=0, hh=0, Tc=30, linear=false)

Args:\n
name (String)\n
ll (Float)\n
omega_c (Vector{Float64}}: (n_waves,)\n
    Central frequencies to perform Taylor expansion\n
n_order (Integer):\n
    Dispersion order\n
kappa (Float): non-linear coefficient (flat)\n

Keyword arguments:\n
ww (Float)\n
    Default = 0 (bulk)\n
hh (Float)\n
    Default = 0 (bulk)\n
tt (Float)\n
    Default = 0 (bulk)\n
angle (Float) \n
    Default = 0 (perpendicular)\n
Tc (Float)\n
    Default = 30 (room temperature)\n
linear (Bool): Is it just a dispersive material?\n
    Default = false
"""
mutable struct TaylorMaterial <: Material
    prop::CommonMaterialProperties
    omega_c::Vector{Float64}
    n_order::Integer
    n_waves::Integer
    beta_array::Matrix{Float64}
    kappa::Float64
    
    function TaylorMaterial(name::String, ll::Float64, omega_c::Vector{Float64}, n_order::Integer, kappa::Float64; 
            ww=0.0, hh=0.0, tt=0.0, angle=0.0, Tc=30.0, linear=false)
        n_waves = length(omega_c)
        beta_array = zeros(max(3, n_order + 1), n_waves)
        new(CommonMaterialProperties(name, ll, ww, hh, tt, angle, Tc, linear), omega_c, n_order, n_waves, beta_array, kappa)
    end
end

function beta_func(mat::TaylorMaterial, omega::Float64, id_wave::Integer)
    return beta_taylor(omega, mat.beta_array[:,id_wave])
end


"""
SellmeierMaterial(name, ll, Avec, Bvec, kappa; ww=0, hh=0, Tc=30, linear=false)

n^2 = 1 + sum_{j} A_j*lbd^2/ (lbd^2 - B_j)

Args:\n
name (String)\n
ll (Float)\n
Avec (Vector{Float64}}: (n,)\n
    Numerator coefficients in Sellmeier's expression
Bvec (Vector{Float64}}: (n,)\n
    Denominator coefficients in Sellmeier's expression
kappa (Float): non-linear coefficient (flat)\n

Keyword arguments:\n
ww (Float)\n
    Default = 0 (bulk)\n
hh (Float)\n
    Default = 0 (bulk)\n
tt (Float)\n
    Default = 0 (bulk)\n
angle (Float) \n
    Default = 0 (perpendicular)\n
Tc (Float)\n
    Default = 30 (room temperature)\n
linear (Bool): Is it just a dispersive material?\n
    Default = false
"""
mutable struct SellmeierMaterial <: Material
    prop::CommonMaterialProperties
    Avec::Vector{Float64}
    Bvec::Vector{Float64}
    kappa::Float64
    
    function SellmeierMaterial(name::String, ll::Float64, Avec::Vector{Float64}, Bvec::Vector{Float64}, kappa::Float64; 
            ww=0.0, hh=0.0, tt=0.0, angle=0.0, Tc=30.0, linear=false)
        new(CommonMaterialProperties(name, ll, ww, hh, tt, angle, Tc, linear), Avec, Bvec, kappa)
    end
end

function beta_func(mat::SellmeierMaterial, omega::Float64)
    return beta_sellmeier(omega, mat.Avec, mat.Bvec)
end

"""
SellmeierMaterial_2
A modified Sellmeier form, based on the work by Jundt, 2008
"""
mutable struct SellmeierMaterial_2 <: Material
    prop::CommonMaterialProperties
    Avec::Vector{Float64}
    Bvec::Vector{Float64}
    kappa::Float64
    funcT
    
    function SellmeierMaterial_2(name::String, ll::Float64, Avec::Vector{Float64}, Bvec::Vector{Float64}, 
            kappa::Float64, funcT;
            ww=0.0, hh=0.0, tt=0.0, angle=0.0, Tc=30.0, linear=false)
        new(CommonMaterialProperties(name, ll, ww, hh, tt, angle, Tc, linear), Avec, Bvec, kappa, funcT)
    end
end

"""
Full dispersion in modified Sellmeier form
"""
function beta_sellmeier_2(omega::Float64, fT::Float64, Avec::Vector{Float64}, Bvec::Vector{Float64})
    k0 = (omega * 1e12/ c0) * 1e-6
    lbd = 2 * pi / k0
    nsq = (Avec[1] + Bvec[1] * fT) - (Avec[end] + Bvec[end] * fT) * lbd^2
    num_vec = Avec[2:2:end-1] .+ Bvec[2:2:end-1] .* fT
    den_vec = Avec[3:2:end-1] .+ Bvec[3:2:end-1] .* fT
    
    n0 = sqrt(nsq + sum(num_vec ./ ((lbd^2) .- den_vec.^2)))
    beta = n0 * k0 * 1e3
    return beta
end;

function beta_func(mat::SellmeierMaterial_2, omega::Float64)
    fT = mat.funcT(mat.prop.Tc)
    return beta_sellmeier_2(omega, fT, mat.Avec, mat.Bvec)
end

"""
RationalMaterial(name, ll, Avec, Bvec, kappa; ww=0, hh=0, Tc=30, linear=false)

Args:\n
name (String)\n
ll (Float)\n
Avec (Vector{Float64}}: (n,)\n
    Numerator coefficients in rational expression
Bvec (Vector{Float64}}: (n,)\n
    Denominator coefficients in rational expression
kappa (Float): non-linear coefficient (flat)\n

Keyword arguments:\n
ww (Float)\n
    Default = 0 (bulk)\n
hh (Float)\n
    Default = 0 (bulk)\n
tt (Float)\n
    Default = 0 (bulk)\n
angle (Float) \n
    Default = 0 (perpendicular)\n
Tc (Float)\n
    Default = 30 (room temperature)\n
linear (Bool): Is it just a dispersive material?\n
    Default = false
"""
mutable struct RationalMaterial <: Material
    prop::CommonMaterialProperties
    Avec::Vector{Float64}
    Bvec::Vector{Float64}
    kappa::Float64
    
    function RationalMaterial(name::String, ll::Float64, Avec::Vector{Float64}, Bvec::Vector{Float64}, kappa::Float64; 
            ww=0.0, hh=0.0, tt=0.0, angle=0.0, Tc=30.0, linear=false)
        new(CommonMaterialProperties(name, ll, ww, hh, tt, angle, Tc, linear), Avec, Bvec, kappa)
    end
end

function beta_func(mat::RationalMaterial, omega::Float64)
    return beta_rational(omega, mat.Avec, mat.Bvec)
end


"""
LaurentMaterial(name, ll, Cvec, Pvec, kappa; ww=0, hh=0, Tc=30, linear=false)

Args:\n
name (String)\n
ll (Float)\n
Cvec (Vector{Float64}}: (n,)\n
    Coefficients in Laurent's equation
Pvec (Vector{Float64}}: (n,)\n
    Denominator coefficients in Sellmeier's expression
kappa (Float): non-linear coefficient (flat)\n

Keyword arguments:\n
ww (Float)\n
    Default = 0 (bulk)\n
hh (Float)\n
    Default = 0 (bulk)\n
tt (Float)\n
    Default = 0 (bulk)\n
angle (Float) \n
    Default = 0 (perpendicular)\n
Tc (Float)\n
    Default = 30 (room temperature)\n
linear (Bool): Is it just a dispersive material?\n
    Default = false
"""
mutable struct LaurentMaterial <: Material
    prop::CommonMaterialProperties
    Cvec::Vector{Float64}
    Pvec::Vector{Float64}
    kappa::Float64
    
    function LaurentMaterial(name::String, ll::Float64, Cvec::Vector{Float64}, Pvec::Vector{Float64}, kappa::Float64; 
            ww=0.0, hh=0.0, tt=0.0, angle=0.0, Tc=30.0, linear=false)
        new(CommonMaterialProperties(name, ll, ww, hh, tt, angle, Tc, linear), Cvec, Pvec, kappa)
    end
end

function beta_func(mat::LaurentMaterial, omega::Float64)
    return beta_laurent(omega, mat.Cvec, mat.Pvec)
end

"""
n_func(mat::Material, omega)
Refractive index
"""
function n_func(mat::Material, omega::Float64)
    k0 = (omega * 1e12) / (c0 * 1e3)
    return beta_func(mat, omega) / k0
end

"""
beta_1_func(mat::Material, omega; d_omega = 1e-3)
"""
function beta_1_func(mat::Material, omega::Float64; d_omega = 1e-3)
    beta_1 = (beta_func(mat, omega + d_omega/2) - beta_func(mat, omega - d_omega/2))/(d_omega)
    return beta_1
end

"""
beta_2_func(mat::Material, omega; d_omega = 1e-3)
"""
function beta_2_func(mat::Material, omega::Float64; d_omega = 1e-3)
    beta_2 = (beta_func(mat, omega + d_omega) + beta_func(mat, omega - d_omega) - 2 * beta_func(mat, omega))/(d_omega^2)
    return beta_2
end

"""
beta_3_func(mat::Material, omega; d_omega = 1e-3)
"""
function beta_3_func(mat::Material, omega::Float64; d_omega = 1e-3)
    beta_3 = (beta_func(mat, omega + 2 * d_omega) - 2 * beta_func(mat, omega + d_omega) + 
        2 * beta_func(mat, omega - d_omega) - beta_func(mat, omega - 2 * d_omega))/(2 * d_omega^3)
    return beta_3
end

"""
beta_4_func(mat::Material, omega; d_omega = 1e-3)
"""
function beta_4_func(mat::Material, omega::Float64; d_omega = 1e-3)
    beta_4 = (beta_func(mat, omega + 2 * d_omega) - 4 * beta_func(mat, omega + d_omega) + 6 * beta_func(mat, omega) - 
        4 * beta_func(mat, omega - d_omega) + beta_func(mat, omega - 2 * d_omega))/(d_omega^4)
    return beta_4
end

"""
convert_to_TaylorMaterial(mat::Material, newmat::TaylorMaterial, omega_c::Vector{Float64}; d_omega = 1e-3, n_order = 2)
Change full dispersion into Taylor-series dispersion
- Check the limit for narrow band processes
"""
function convert_to_TaylorMaterial(mat::Material, newmat::TaylorMaterial, omega_c::Vector{Float64}; d_omega = 1e-3, n_order = 2)
    newmat.prop.ll= mat.prop.ll
    newmat.prop.ww = mat.prop.ww
    newmat.prop.hh = mat.prop.hh
    newmat.prop.tt = mat.prop.tt
    newmat.prop.angle = mat.prop.angle
    newmat.prop.Tc = mat.prop.Tc
    newmat.prop.linear = mat.prop.linear
    
    newmat.omega_c = omega_c
    newmat.n_order = n_order
    newmat.n_waves = length(omega_c)
    newmat.kappa = mat.kappa
    
    if n_order >= 0
       newmat.beta_array[1,:] = [beta_func(mat, omg) for omg in omega_c]
    end
    
    if n_order >= 1
       newmat.beta_array[2,:] = [beta_1_func(mat, omg; d_omega = d_omega) for omg in omega_c]
    end
    
    if n_order >= 2
       newmat.beta_array[3,:] = [beta_2_func(mat, omg; d_omega = d_omega) for omg in omega_c]
    end
    
    if n_order >= 3
       newmat.beta_array[4,:] = [beta_3_func(mat, omg; d_omega = d_omega) for omg in omega_c]
    end
    
    if n_order >= 4
       newmat.beta_array[5,:] = [beta_4_func(mat, omg; d_omega = d_omega) for omg in omega_c]
    end
    
    if n_order >= 5
        throw(DomainError(5, "Dispersion order is not implemented!"))
    end
end

# SETTING THE PARAMETERS INTERNALLY
export set_beta_array, set_length, set_width, set_height, set_thickness, set_angle, set_temperature, set_kappa

function set_beta_array(mat::TaylorMaterial, n_order::Integer, beta_n_vec::Vector{Float64})
    mat.beta_array[n_order,:] = beta_n_vec
end

function set_length(mat::Material, ll::Float64)
    mat.prop.ll = ll
end

function set_width(mat::Material, ww::Float64)
    mat.prop.ww = ww
end

function set_height(mat::Material, hh::Float64)
    mat.prop.hh = hh
end

function set_thickness(mat::Material, tt::Float64)
    mat.prop.tt = tt
end

function set_angle(mat::Material, theta::Float64)
    mat.prop.angle = theta
end

function set_temperature(mat::Material, Tc::Float64)
    mat.prop.Tc = Tc
end

function set_kappa(mat::Material, kappa::Float64)
    mat.kappa = kappa
end

# =======================
# LIBRARY OF COMMON MATERIALS
export air, smf28, si, sio2, si3n4

gayer_tables = [
    [5.756     5.653     5.078  ];
    [0.0983    0.1185    0.0964 ];
    [0.2020    0.2091    0.2065];
    [189.32    89.61     61.16   ];
    [12.52     10.85     10.55   ];
    [1.32e-2   1.97e-2   1.59e-2 ];
    [2.860e-6  7.941e-7  4.677e-6];
    [4.700e-8  3.134e-8  7.822e-8];
    [6.113e-8 -4.641e-9 -2.653e-8];
    [1.516e-4 -2.188e-6  1.096e-4];
]

gayer_ne_mgo5_avec = gayer_tables[1:6,1]
gayer_no_mgo5_avec = gayer_tables[1:6,2]
gayer_ne_mgo1_avec = gayer_tables[1:6,3]

gayer_ne_mgo5_bvec = [gayer_tables[7:10,1]; zeros(2)]
gayer_no_mgo5_bvec = [gayer_tables[7:10,2]; zeros(2)]
gayer_ne_mgo1_bvec = [gayer_tables[7:10,3]; zeros(2)]


air = ConstantMaterial("air", 1.0, 1.0, 0.0; linear=true)
smf28 = LaurentMaterial("smf28", 1.0, [1.455, -0.003225, 0.003195], [0., 2., -2.], 0.0)
si = SellmeierMaterial("Si", 1.0, [10.6684, 0.003043, 1.5413], [0.3015, 1.1347, 1104].^2, 0.0)
sio2 = SellmeierMaterial("SiO2", 1.0, [0.6961, 0.4079, 0.8974], [0.06840, 0.1162, 9.8961].^2, 0.0)
si3n4 = SellmeierMaterial("Si3N4", 1.0, [2.8939], [0.1396].^2, 0.0)

export linbo3_e_jundt, linbo3_i_jundt, linbo3_e_gayer, linbo3_o_gayer, linbo3_e_gayer_mgo1
export lnlt_z, lnoi_z

linbo3_e_jundt = SellmeierMaterial("LiNbO3 e-pol - Jundt 1997", 1.0, [2.9804, 0.5981, 8.9543], [0.02047, 0.0666, 416.08], 0.0)
linbo3_o_jundt = SellmeierMaterial("LiNbO3 o-pol - Jundt 1997", 1.0, [2.6734, 1.2290, 12.614], [0.01764, 0.05914, 474.60], 0.0)

fT_gayer(T) = (T - 24.5)*(T + 570.82)
linbo3_e_gayer = SellmeierMaterial_2("LiNbO3 e-pol (5% MgO) - Gayer 2008", 
    1.0, gayer_ne_mgo5_avec, gayer_ne_mgo5_bvec, 0.0, fT_gayer; Tc = 24.5)
linbo3_o_gayer = SellmeierMaterial_2("LiNbO3 e-pol (5% MgO) - Gayer 2008", 
    1.0, gayer_no_mgo5_avec, gayer_no_mgo5_bvec, 0.0, fT_gayer; Tc = 24.5)
linbo3_e_gayer_mgo1 = SellmeierMaterial_2("LiNbO3 e-pol (1% MgO) - Gayer 2008", 
    1.0, gayer_ne_mgo1_avec, gayer_ne_mgo1_bvec, 0.0, fT_gayer; Tc = 24.5)

# Waveguide simulation (Rational form)
Avec_lnlt_z = npzread("/home/elaksono/eo-comb/SplitStep/dispersion/Avec_33_LNLT_Z_w8000_y8000_h8000_th0.1_Tc30.npy");
Bvec_lnlt_z = npzread("/home/elaksono/eo-comb/SplitStep/dispersion/Bvec_33_LNLT_Z_w8000_y8000_h8000_th0.1_Tc30.npy");
lnlt_z = RationalMaterial("NTT: LiNbO3 on LiTaO3", 30.0, Avec_lnlt_z, Bvec_lnlt_z, 0.0; 
    ww = 8e-3, tt = 8e-3, hh = 8e-3, angle = 0.1)

Avec_lnoi_z = npzread("/home/elaksono/eo-comb/SplitStep/dispersion/Avec_33_LNOI_Z_w5000_y7500_h4500_th30_Tc30.npy");
Bvec_lnoi_z = npzread("/home/elaksono/eo-comb/SplitStep/dispersion/Bvec_33_LNOI_Z_w5000_y7500_h4500_th30_Tc30.npy");
lnoi_z = RationalMaterial("HCP: LiNbO3 on SiO2", 30.0, Avec_lnoi_z, Bvec_lnoi_z, 0.0; 
    ww = 5e-3, tt = 7.5e-3, hh = 4.5e-3, angle = 30.0)

end