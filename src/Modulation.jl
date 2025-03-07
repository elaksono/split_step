module Modulation
export fm_sin2, fm_cos2, fm_cos, fm_sin, modulation_func

# Time modulation
fm_sin(t, T, A, ϕ) = A * sin(2*pi*t/T + ϕ)
fm_cos(t, T, A, ϕ) = A * cos(2*pi*t/T + ϕ)
fm_sin2(t, T, A, ϕ) = 2 * A * sin(pi*t/T + ϕ)^2
fm_cos2(t, T, A, ϕ) = 2 * A * cos(pi*t/T + ϕ)^2

# The function form: f(t, T, A, ϕ)
modulation_func = Dict("sin" => fm_sin, "cos" => fm_cos, "sin2" => fm_sin2, "cos2" => fm_cos2)

# Nonlinear coefficient - spatial modulation
# Poling profile for a nonlinear process
export phi_z_con, phi_z_lin, phase_z_func
phi_z_con(z, k0) = k0 * z
phi_z_lin(z, k0, zc, dk_1) = (dk_1/2) * ((z-zc)^2 - zc^2) + k0 * z

phase_z_func = Dict("con" => phi_z_con, "lin" => phi_z_lin)

# Phase mismatch
export dk_z_lin
dk_z_lin(z, k0, zc, dk_1) = dk_1 * (z-zc) + k0

# Tapered gain
export g_con, g_tanh, taper_func
g_con(z) = 1.0
g_tanh(z, L, w) = tanh(z/w) * tanh((L-z)/w)

taper_func = Dict("con" => g_con, "tanh" => g_tanh)

end