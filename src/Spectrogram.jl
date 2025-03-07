using Plots, FFTW

# Function to generate a spectrogram using FFTW
function plot_spectrogram(signal::Vector{<:Real}, M::Int; window_type::Symbol = :rectangular, fs::Real = 1.0, hop_size::Integer = 0)
    N = length(signal)
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
    num_segments = 1 + div(N - M, hop_size)

    # Initialize matrix to hold FFT results
    spectrogram_data = zeros(Complex{Float64}, M ÷ 2 + 1, num_segments)

    # Loop over segments and compute FFT
    for i in 1:num_segments
        start_idx = (i - 1) * hop_size + 1
        segment = signal[start_idx:start_idx + M - 1] .* window
        fft_result = fft(segment)

        # Store only the positive frequencies
        spectrogram_data[:, i] = fft_result[1:M ÷ 2 + 1]
    end

    # Get magnitude squared to compute the spectrogram
    spectrogram_data = abs2.(spectrogram_data)

    # Plot the spectrogram
    time_vec = range(0, (num_segments - 1) * hop_size / fs, length=num_segments)
    freq_vec = range(0, fs / 2, length=M ÷ 2 + 1)

    plot(
        time_vec,
        freq_vec,
        spectrogram_data,
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
        title="Spectrogram",
        seriestype=:heatmap,
        color=:viridis
    )
end