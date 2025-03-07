module PlotPulse

using LaTeXStrings, Plots, FFTW

export calculate_spectrogram, plot_spectrogram

# Function to generate a spectrogram using FFTW
function calculate_spectrogram(input_signal::Union{Vector{<:Real}, Vector{<:Complex}}, M::Int; window_type::Symbol = :rectangular,
        fs::Real = 1.0, hop_size::Integer = 0, cyclical::Bool = true, freq_ctr = 0.0, arr_time = nothing, arr_omega = nothing)
    N = length(input_signal)
    signal = cyclical ? [input_signal[end - M ÷ 2 + 1: end]; input_signal; input_signal[1:M ÷ 2]] : input_signal
    
    if hop_size == 0
        hop_size = M ÷ 2  # Overlap of 50%
    end
        
    # Define the window
    if window_type == :gaussian
        window = exp.(-0.5 .* ((collect(0:M-1) .- M/2) ./ (M/6)).^2)
    else
        window = ones(M)  # Rectangular window
    end

    # Normalize window power
    window /= sqrt(sum(window.^2) / M)

    # Compute number of segments
    num_segments = cyclical ? div(N, hop_size) : 1 + div(N - M, hop_size)

    # Initialize matrix to hold FFT results
    spectrogram_data = zeros(Complex{Float64}, M, num_segments)

    # Loop over segments and compute FFT
    for i in 1:num_segments
        start_idx = (i - 1) * hop_size + 1
        segment = signal[start_idx:start_idx + M - 1] .* window       
        fft_result = fft(segment)

        # Store the full FFT result (including negative frequencies)
        spectrogram_data[:, i] = fftshift(fft_result)
    end

    # Get magnitude squared to compute the spectrogram
    spectrogram_data = abs2.(spectrogram_data)

    # Plot the spectrogram
    if isnothing(arr_time)
        time_vec = range(0, (num_segments - 1) * hop_size / fs, length=num_segments)
    else
        time_vec = sort(arr_time)
        time_vec = time_vec[1:num_segments]
    end
    
    if isnothing(arr_omega)
        freq_vec = range(-fs / 2, stop = fs / 2, length = M) .+ freq_ctr
    else
        freq_vec = sort(arr_omega) ./ (2*pi)
        D_freq = length(freq_vec) ÷ M
        freq_vec = freq_vec[1:D_freq:end]
    end
    
    output = Dict("time" => time_vec, "freq" => freq_vec, "spectrogram" => spectrogram_data)
    return output
end

function plot_spectrogram(dict_spectrogram::Dict; x_unit = "(ps)", y_unit = "(THz)", 
        y_axis = "freq", title = "Spectrogram", color=:viridis, plot_holder = nothing)
    x_data = dict_spectrogram["time"]
    freq_vec = dict_spectrogram["freq"]
    lbd_vec = (3e8 * 1e6) ./ (1e12 .* freq_vec)
    spectrogram_data = dict_spectrogram["spectrogram"]
    
    if (y_axis == "freq")
        y_data = freq_vec
        y_label = "Frequency " * y_unit
    else
        y_data = lbd_vec
        id_y = sortperm(y_data)
        y_data = y_data[id_y]
        spectrogram_data = spectrogram_data[id_y,:]
        y_label = "Wavelength " * y_unit
    end
    
    if isnothing(plot_holder) plot_holder = plot() end
    plot!(plot_holder,
        x_data, 
        y_data,
        spectrogram_data,
        xlabel="Time " * x_unit,
        ylabel=y_label,
        title=title,
        seriestype=:heatmap,
        color=color)
    
    return plot_holder
end


export plotCompare
"""
plotCompare(arr_x, num_pulse, anl_pulse, plot_var;\n
        dom = "t", xtext = "Time", comp = real, imax = nothing, ylabel = "Re[A(t)]",
        legend=:bottomleft, margins=5Plots.mm)

Plot to compare the pulse profile obtained from numerical simulation and analytical solution
arr_x can be an array of time or frequency points
comp can be real, imag, or abs
"""
function plotCompare(arr_x, num_pulse, anl_pulse, plot_var; 
        domain = "t", xtext = "Time", comp = real, imax = nothing, ylabel = "Re[A(t)]", color = nothing,
        palette = palette(:tab10), legend=:bottomleft, margins=5Plots.mm)
    if imax == nothing
        imax = size(num_pulse)[2]
    end
    
    sorted_id = sortperm(arr_x)
    sorted_x = arr_x[sorted_id]
    
    for i in collect(1:imax)
        if color == nothing
            color = palette[i]
        end
        
        scatter!(plot_var, arr_x, comp.(num_pulse[:,i]), label = "Numeric " * L"A_{%$i}(%$domain)", color = color,
            xlabel = xtext * " " * domain, ylabel = ylabel, legend=legend, left_margin=5margins, bottom_margin=5margins)
        plot!(plot_var, sorted_x, comp.(anl_pulse[sorted_id,i]), label = "Analytic " * L"A_{%$i}(%$domain)", color = color)
    end
end;

end