module Modulation
export fm_sin2, fm_cos2, fm_cos, fm_sin

fm_sin(t, T, A, ϕ) = A * sin(2*pi*t/T + ϕ)
fm_cos(t, T, A, ϕ) = A * cos(2*pi*t/T + ϕ)
fm_sin2(t, T, A, ϕ) = A * sin(pi*t/T + ϕ)^2
fm_cos2(t, T, A, ϕ) = A * cos(pi*t/T + ϕ)^2

end