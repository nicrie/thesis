# %%
import matplotlib.pyplot as plt
import numpy as np

# Set FFT length (the longer this is, the closer we get to the ideal filter)
N = 1024

# Create a frequency grid for -pi to pi
# (We use fftshift conventions so that the zero frequency is at the center.)
w = np.linspace(-np.pi, np.pi, N, endpoint=False)

# Ideal Hilbert transform multiplier in the frequency domain:
# H(ω) = -j * sgn(ω)
H_ideal = -1j * np.sign(w)
# (At ω = 0, sgn(0)=0; this is the theoretical definition.)

# Get the corresponding impulse response by taking the inverse FFT.
# We first use ifftshift to put the zero frequency back at index 0.
h_ideal = np.fft.ifft(np.fft.ifftshift(H_ideal))
# Since the ideal Hilbert transformer is a real, odd function in time,
# we take the real part (small imaginary numerical errors may appear).
h_ideal = np.real(h_ideal)
# Shift the impulse response so that time zero is in the middle.
h_ideal = np.fft.fftshift(h_ideal)

# Plot the (infinite-duration) ideal impulse response (truncated to N samples)
plt.figure(figsize=(10, 4))
plt.stem(np.arange(-N // 2, N // 2), h_ideal, basefmt=" ")
plt.xlabel("Sample index")
plt.ylabel("Amplitude")
plt.title("Impulse Response of the Ideal Hilbert Transformer (Finite FFT length)")
plt.grid(True)
plt.show()

# --- Now, simulate the effect of windowing (i.e. truncating the impulse response) ---

# Let’s choose a shorter impulse response length (a rectangular window).
M = 129  # M << N; you can change M to see different leakage effects
start = (N - M) // 2  # center the window within the full impulse response

# Create a truncated impulse response by zeroing out samples outside the window.
h_windowed = np.zeros_like(h_ideal)
h_windowed[start : start + M] = h_ideal[start : start + M]

# Compute the frequency response of the windowed (truncated) Hilbert transformer.
H_windowed = np.fft.fftshift(np.fft.fft(h_windowed, N))
# Create a frequency axis in radians/sample
# (Note: w here goes from -pi to pi, matching our earlier grid.)
freq_axis = w

# Plot the magnitude and phase of the windowed filter’s frequency response.
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(freq_axis, np.abs(H_windowed), "b")
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Magnitude")
plt.title("Magnitude Response of the Windowed Hilbert Transformer")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(freq_axis, np.angle(H_windowed), "r")
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Phase (radians)")
plt.title("Phase Response of the Windowed Hilbert Transformer")
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Explanation ---
# In an ideal Hilbert transformer, the frequency response should be an abrupt jump (discontinuity)
# in phase: -90° for positive frequencies and +90° for negative frequencies.
# However, by truncating the impulse response (i.e. using a finite-length filter), the sharp
# transition is smeared out. This “smearing” appears as ripples (oscillations) in the magnitude
# and phase plots – a clear example of spectral leakage due to windowing.

# %%
# Compare 2 signals: low and high spectral leakage
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

# Parameters
Fs = 1000  # Sampling frequency in Hz
T = 1.0  # Signal duration in seconds
N = int(Fs * T)  # Number of samples
t = np.linspace(0, T, N, endpoint=False)

# Create two signals:
# 1. Low spectral leakage: sine wave with an integer number of cycles (10 Hz -> exactly 10 cycles in 1 sec)
f_low = 10  # Frequency in Hz (integer number of cycles in T seconds)
signal_low = np.sin(2 * np.pi * f_low * t)

# 2. High spectral leakage: sine wave with a non-integer number of cycles (10.3 Hz -> 10.3 cycles in 1 sec)
f_high = 10.3  # Frequency in Hz (non-integer cycles)
signal_high = np.sin(2 * np.pi * f_high * t)

# Compute analytic signals using the Hilbert transform.
# The imaginary part of the analytic signal is the Hilbert transform.
analytic_low = hilbert(signal_low)
analytic_high = hilbert(signal_high)

# Compute amplitude envelopes (magnitude of the analytic signal)
envelope_low = np.abs(analytic_low)
envelope_high = np.abs(analytic_high)

# Compute FFTs of the analytic signals (for frequency-domain visualization)
fft_low = np.fft.fftshift(np.fft.fft(analytic_low))
fft_high = np.fft.fftshift(np.fft.fft(analytic_high))
freq = np.linspace(-Fs / 2, Fs / 2, N, endpoint=False)

# Plotting: create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 8))

# Time domain: Low spectral leakage signal
axs[0, 0].plot(t, signal_low, label="Signal (10 Hz)", color="C0")
axs[0, 0].plot(t, envelope_low, "--r", label="Envelope")
axs[0, 0].set_title("Low Leakage: Integer Cycles (10 Hz)")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].set_ylabel("Amplitude")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Frequency domain: Low spectral leakage signal
axs[0, 1].plot(freq, np.abs(fft_low) / N, color="C0")
axs[0, 1].set_title("FFT of Analytic Signal (Low Leakage)")
axs[0, 1].set_xlabel("Frequency [Hz]")
axs[0, 1].set_ylabel("Magnitude")
axs[0, 1].grid(True)

# Time domain: High spectral leakage signal
axs[1, 0].plot(t, signal_high, label="Signal (10.3 Hz)", color="C1")
axs[1, 0].plot(t, envelope_high, "--r", label="Envelope")
axs[1, 0].set_title("High Leakage: Non-integer Cycles (10.3 Hz)")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_ylabel("Amplitude")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Frequency domain: High spectral leakage signal
axs[1, 1].plot(freq, np.abs(fft_high) / N, color="C1")
axs[1, 1].set_title("FFT of Analytic Signal (High Leakage)")
axs[1, 1].set_xlabel("Frequency [Hz]")
axs[1, 1].set_ylabel("Magnitude")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

# %%


def transform_hilbert(data, ext="none", alpha=1):
    N, D = data.shape
    if ext == "none":
        analytic_data = hilbert(data, axis=0)
    elif ext == "zero":
        # Pad with zeros: pad N rows before and after along axis=0.
        pad_width = ((N, N), (0, 0))
        data_padded = np.pad(data, pad_width, mode="constant", constant_values=0)
        # Apply the Hilbert transform along the time axis (axis 0)
        analytic_data_padded = hilbert(data_padded, axis=0)
        # Extract the central region corresponding to the original data
        analytic_data = analytic_data_padded[N : 2 * N, :]
    elif ext == "constant":
        # Prepare padded array of shape (3 * N, D)
        data_padded = np.empty((3 * N, D), dtype=data.dtype)

        # Left pad: fill with the first row (each column's first value)
        data_padded[:N, :] = np.repeat(data[0:1, :], N, axis=0)
        # Middle: original data
        data_padded[N : 2 * N, :] = data
        # Right pad: fill with the last row (each column's last value)
        data_padded[2 * N :, :] = np.repeat(data[-1:, :], N, axis=0)

        # Apply the Hilbert transform along the time axis on the padded data
        analytic_data_padded = hilbert(data_padded, axis=0)
        # Extract the central portion corresponding to the original data size
        analytic_data = analytic_data_padded[N : 2 * N, :]
    elif ext == "exp_zero":
        # Generate left pad weights: from far left to the boundary.
        # For i=0 (far left) weight = exp(-alpha*(n_samples-1)), for i=n_samples-1 weight = exp(0)=1.
        left_weights = np.exp(-alpha * np.arange(N - 1, -1, -1)).reshape(N, 1)
        # Generate right pad weights: for j=0 (adjacent to data) weight = exp(0)=1, for j=N-1 weight = exp(-alpha*(N-1)).
        right_weights = np.exp(-alpha * np.arange(N)).reshape(N, 1)

        # Create left and right pads for each feature by broadcasting the weights.
        left_pad = data[0:1, :] * left_weights  # shape: (N, n_features)
        right_pad = data[-1:, :] * right_weights  # shape: (N, n_features)

        # Concatenate the pads with the original data.
        data_padded = np.concatenate([left_pad, data, right_pad], axis=0)

        # Apply Hilbert transform along the time axis (axis 0).
        analytic_data_padded = hilbert(data_padded, axis=0)

        # Extract the central portion corresponding to the original data.
        analytic_data = analytic_data_padded[N : 2 * N, :]
    elif ext == "exp_lin":
        x = np.arange(N)
        x_ext = np.arange(-x.size, 2 * x.size)

        coefs = np.polynomial.polynomial.polyfit(x, data, deg=1)
        yfit = np.polynomial.polynomial.polyval(x, coefs).T
        yfit_ext = np.polynomial.polynomial.polyval(x_ext, coefs).T

        y_ano = data - yfit

        amp_pre = np.take(y_ano, 0, axis=0)[:, None]
        amp_pos = np.take(y_ano, -1, axis=0)[:, None]

        exp_ext = np.exp(-alpha * x)
        exp_ext_reverse = exp_ext[::-1]

        pad_pre = amp_pre * exp_ext_reverse
        pad_pos = amp_pos * exp_ext

        y_ext = np.concatenate([pad_pre.T, y_ano, pad_pos.T], axis=0)
        y_ext += yfit_ext

        data_padded = y_ext

        # Apply Hilbert transform along the time axis (axis 0).
        analytic_data_padded = hilbert(data_padded, axis=0)

        # Extract the central portion corresponding to the original data.
        analytic_data = analytic_data_padded[N : 2 * N, :]
    else:
        ValueError("stupid!")

    analytic_data = analytic_data - analytic_data.mean(0)
    return analytic_data


# %%
# Synthetic data
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert

n_features = 10


def create_synthetic_dataset(
    leakage_param=0.0, noise_level=0.1, trend_amp=0.0, base_amp=1.0, n_features=10
):
    # ---------------------------
    # Parameters for the dataset
    # ---------------------------
    Fs = 1000.0  # Sampling frequency in Hz
    T = 1.0  # Total duration in seconds
    N = int(Fs * T)  # Number of time samples
    t = np.linspace(0, T, N, endpoint=False)

    # ---------------------------
    # Tunable parameters
    # ---------------------------
    # 1. Spectral leakage tuning:
    #    leakage_param = 0 -> sine wave with integer # of cycles (minimal leakage)
    #    leakage_param = 1 -> sine wave with non-integer # of cycles (maximal leakage)
    # leakage_param = 0
    f_base = 10.0  # Base frequency in Hz (yields integer cycles if leakage_param==0)
    f_offset = 0.3  # Frequency offset added when leakage_param > 0
    f_signal = f_base + leakage_param * f_offset

    # 2. White noise level (standard deviation)
    # noise_level = 0.1

    # 3. Trend amplitude (linear trend added to all channels)
    # trend_amp = 0

    # 4. Base amplitude for the sinusoidal signals
    # base_amp = 1.0

    # ---------------------------
    # Spatial Gaussian profile
    # ---------------------------
    # Create a spatial weighting that is strongest in the center and decays toward the edges.
    x = np.arange(n_features)
    center = (n_features - 1) / 2.0
    sigma = n_features / 4.0  # Adjust sigma to control the spread of the Gaussian
    spatial_weights = np.exp(-((x - center) ** 2) / (2 * sigma**2))
    # Normalize weights if desired (here maximum amplitude remains base_amp)
    # spatial_weights = spatial_weights / np.max(spatial_weights)

    # ---------------------------
    # Generate the synthetic dataset
    # ---------------------------
    # The dataset will have dimensions: (time samples, spatial features)
    data = np.zeros((N, n_features))

    for i in range(n_features):
        # Random phase for each channel
        phase = 2 * np.pi * np.random.rand()
        # Modulate the amplitude by the spatial weight
        amp = base_amp * spatial_weights[i]
        # Create the sinusoidal signal
        sinusoid = amp * np.sin(2 * np.pi * f_signal * t + phase)
        # Add a linear trend (same for all channels)
        trend = trend_amp * t
        # Add white noise
        noise = noise_level * np.random.randn(N)
        # Combine components
        data[:, i] = sinusoid + trend + noise

    # Zero mean time series
    data -= data.mean(0)
    return pd.DataFrame(data, index=t)


def compute_psd_fft(x, fs):
    """
    Compute PSD of a signal using a full-spectrum FFT.

    Parameters
    ----------
    x : ndarray
        Time-series of length N.
    fs : float
        Sampling frequency.

    Returns
    -------
    freqs : ndarray
        Frequency bins in ascending order from 0 to +fs/2.
    psd : ndarray
        Power spectral density, same length as freqs.
    """
    N = len(x)

    # Compute the full DFT
    X = np.fft.fft(x)

    # PSD = magnitude-squared / (N*fs) (one common convention)
    psd = (np.abs(X) ** 2) / (N * fs)

    # One-sided power (for real signals)
    P1 = psd[: N // 2 + 1]
    P1[1:-1] *= 2  # because we’re folding negative frequencies onto positive
    freqs = np.linspace(0, fs / 2, len(P1))

    return freqs, P1


nl = 0.2
data1 = create_synthetic_dataset(leakage_param=0, trend_amp=0, noise_level=nl)
data2 = create_synthetic_dataset(leakage_param=1, trend_amp=0, noise_level=nl)
data3 = create_synthetic_dataset(
    leakage_param=1, trend_amp=1, base_amp=1, noise_level=nl
)


data2.iloc[:, 5].plot()
# %%
data = data3
x = data.index.values
Y = data.values
# ---------------------------
# Apply the Hilbert Transform
# ---------------------------
# Since our time axis is axis=0 (rows), we apply the Hilbert transform along that axis.
paddings = ["none", "zero", "constant", "exp_zero", "exp_lin"]

for pad in paddings:
    print(pad)
    Ya = transform_hilbert(Y, ext=pad, alpha=1e-1)

    # ---------------------------
    # Visualization for one selected channel (e.g., the center channel)
    # ---------------------------
    selected_channel = n_features // 2

    # Extract the analytic signal and its envelope (magnitude)
    analytic_signal = Ya[:, selected_channel]
    envelope = np.abs(analytic_signal)

    # Plot the time-domain signal and its envelope
    plt.figure(figsize=(12, 5))
    plt.plot(x, Y[:, selected_channel], label="Original Signal")
    plt.plot(x, envelope, "--r", label="Envelope (Hilbert)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Channel {selected_channel}, Padding {pad}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------------
    # Frequency-Domain Visualization
    # ---------------------------
    # Compute the FFT of the analytic signal for the selected channel
    # fft_analytic = np.fft.fftshift(np.fft.fft(analytic_signal))
    # freq_axis = np.linspace(-Fs / 2, Fs / 2, N, endpoint=False)

    # plt.figure(figsize=(12, 5))
    # plt.plot(freq_axis, np.abs(fft_analytic) / N)
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Magnitude")
    # plt.title("FFT of the Analytic Signal (Selected Channel)")
    # plt.grid(True)
    # plt.show()

    # ---------------------------
    # Final Notes:
    # ---------------------------
    # The generated dataset "data" has dimensions (time samples x spatial features) and contains:
    # - A sinusoidal signal with tunable spectral leakage (via f_signal controlled by leakage_param)
    # - A linear trend (controlled by trend_amp)
    # - Additive white noise (controlled by noise_level)
    # - Spatial amplitude modulation via a Gaussian profile (spatial_weights)
    #
    # The analytic_data (obtained by the Hilbert transform) can now be used for further analysis,
    # such as Hilbert PCA, to study the impact of spectral leakage.

    # ---------------------------
    # Next Steps: Hilbert PCA
    # ---------------------------
    # With your synthetic spatio-temporal dataset 'data' and its Hilbert-transformed version
    # 'analytic_data', you can now perform PCA on the complex (analytic) data to study how
    # spectral leakage might affect the extracted spatio-temporal modes.
    #
    # For example, you might construct a covariance matrix from the analytic signals,
    # perform an eigen-decomposition, and compare the modes for different values of leakage_param.
    U, s, VT = np.linalg.svd(Ya, full_matrices=False)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3))

    ax[0].plot(U.real[:, 0])
    ax[0].plot(U.imag[:, 0])
    ax[0].plot(np.abs(U)[:, 0], color=".3")

    ax[1].plot(U[:, 1].real)
    ax[1].plot(U[:, 1].imag)
    ax[1].plot(np.abs(U)[:, 1], color=".3")
    plt.show()

# %%
