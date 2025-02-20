# %%
import numpy as np
import seaborn as sns
from scipy.signal import hilbert

sns.set_context("paper", font_scale=0.8)


def compute_psd_fft(x, fs):
    N = len(x)
    # Perform FFT
    X = np.fft.fft(x)
    # Compute two-sided power
    P2 = np.abs(X) ** 2 / (N * fs)
    # One-sided power (for real signals)
    P1 = P2[: N // 2 + 1]
    P1[1:-1] *= 2  # because we’re folding negative frequencies onto positive
    freqs = np.linspace(0, fs / 2, len(P1))
    return freqs, P1


def generate_sinusoids(N, fs, freq=5.0, amplitude=1.0, phase=0.0):
    """
    Generate a sinusoid of length N at given freq, amplitude, and phase.
    """
    t = np.arange(N) / fs
    return amplitude * np.sin(2.0 * np.pi * freq * t + phase)


def generate_linear_trend(N, slope=0.01, intercept=0.0):
    """
    Generate a linear ramp from 0..N-1 with given slope + intercept.
    """
    return slope * np.arange(N) + intercept


def generate_ar1(N, alpha=0.9, std=1.0, seed=None):
    """
    Generate an AR(1) process: x[t] = alpha * x[t-1] + eps[t],
    with eps ~ Normal(0, std^2), length N.
    """
    rng = np.random.default_rng(seed)

    x = np.zeros(N, dtype=float)
    for t in range(1, N):
        x[t] = alpha * x[t - 1] + rng.normal(0, std)
    return x


def generate_orthonormal_vectors(dim=10, num_vecs=3, seed=None):
    """
    Generate 'num_vecs' orthonormal vectors in R^dim using a random approach + QR decomposition.
    """
    rng = np.random.default_rng(seed)
    # Start with a random (dim x num_vecs) matrix
    shape = (dim, num_vecs)
    A = rng.standard_normal(size=shape) + 1j * rng.standard_normal(size=shape)
    # Orthonormalize via QR. Q will be (dim x num_vecs), R is (num_vecs x num_vecs).
    Q, R = np.linalg.qr(A)
    # Return the first 'num_vecs' columns (they are already orthonormal).
    return Q[:, :num_vecs]


def construct_synthetic_data(
    N=1000,
    D=10,
    fs=100.0,
    sinus_params=None,
    trend_params=None,
    ar1_params=None,
    seed=None,
):
    """
    Construct a D-dimensional dataset of length N in time, consisting of
    3 distinct components (sinusoidal, linear trend, AR(1)) embedded
    in orthonormal directions.

    Returns
    -------
    data : ndarray of shape (N, D)
        The time x space data matrix.
    sources : dict
        A dictionary with 'sinus', 'trend', and 'ar1' arrays of shape (N,).
    V : ndarray of shape (D, 3)
        The orthonormal vectors used for each of the 3 components.
    """
    rng = np.random.default_rng(seed)

    # 1) Generate the 3 source signals, each of length N
    if sinus_params is None:
        sinus_params = {
            "freq": 5.0,
            "amplitude": 1.0,
        }
    if trend_params is None:
        trend_params = {"slope": 0.01, "intercept": 0.0}
    if ar1_params is None:
        ar1_params = {"alpha": 0.98, "std": 0.5}

    # s_trend = generate_linear_trend(N, **trend_params)
    s_trend_re = generate_sinusoids(N, fs, phase=0, freq=0.04, amplitude=1.8)
    s_trend_im = generate_sinusoids(N, fs, phase=np.pi / 2, freq=0.04, amplitude=1.8)
    s_ar1 = generate_ar1(N, **ar1_params, seed=seed)
    s_sinus_re = generate_sinusoids(N, fs, phase=0, **sinus_params)
    s_sinus_im = generate_sinusoids(N, fs, phase=np.pi / 2, **sinus_params)

    s_trend = s_trend_re + 1j * s_trend_im
    s_sinus = s_sinus_re + 1j * s_sinus_im
    S = np.vstack([s_trend, s_ar1, s_sinus]).T

    # (Optional) you can scale them so they have comparable variances
    # or keep them as is to highlight one vs. another.

    # 2) Generate 3 orthonormal vectors in D-dimensional space
    V = generate_orthonormal_vectors(dim=D, num_vecs=3, seed=seed)
    std_v = 1e-0 * V.std(0).mean()
    Vnoise = V + rng.normal(0, scale=std_v, size=V.shape)

    # 3) Combine the sources in 10D space
    # data[t, :] = s1(t)*v1 + s2(t)*v2 + s3(t)*v3
    data = S @ Vnoise.conj().T
    data_trend_only = S[:, 0:1] @ Vnoise[:, 0:1].conj().T
    data_ar1_only = S[:, 1:2] @ Vnoise[:, 1:2].conj().T
    data_sinus_only = S[:, 2:3] @ Vnoise[:, 2:3].conj().T

    # 4) [Optional] Add small noise in orthogonal subspace or overall
    #    to avoid perfect rank-3 data.
    # For instance:
    noise_strength = 0.02
    data_noise_only = rng.normal(scale=noise_strength, size=data.shape)
    data += data_noise_only

    data = data - data.mean(0)

    sources = {"sinus": s_sinus, "trend": s_trend, "ar1": s_ar1}
    data_decomposed = {
        "sinus": data_sinus_only.real,
        "trend": data_trend_only.real,
        "ar1": data_ar1_only.real,
        "noise": data_noise_only,
    }
    return data.real, sources, V, data_decomposed


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
import matplotlib.pyplot as plt

D = 300
Y, sources, V, Y_dec = construct_synthetic_data(
    D=D,
    seed=123,
    sinus_params={"freq": 1.73, "amplitude": 1.0},
    trend_params={"slope": 4e-3},
)

variance = {
    "sinus": Y_dec["sinus"].var(0).sum(),
    "ar1": Y_dec["ar1"].var(0).sum(),
    "trend": Y_dec["trend"].var(0).sum(),
    "noise": Y_dec["noise"].var(0).sum(),
}
import pandas as pd

var_true = pd.Series(
    variance.values(), index=variance.keys(), name="explained_variance"
)
var_true = var_true.sort_values(ascending=False)

frac_var_true = var_true / var_true.sum()
print(frac_var_true.round(2))

# ---------------------------
# Apply the Hilbert Transform
# ---------------------------
# Since our time axis is axis=0 (rows), we apply the Hilbert transform along that axis.
paddings = ["none", "zero", "constant", "exp_zero", "exp_lin"]

for pad in paddings:
    print(pad)
    Ya = transform_hilbert(Y, ext=pad, alpha=4e-2)

    # ---------------------------
    # Visualization for one selected channel (e.g., the center channel)
    # ---------------------------
    selected_channel = D // 2

    # Extract the analytic signal and its envelope (magnitude)
    analytic_signal = Ya[:, selected_channel]
    envelope = np.abs(analytic_signal)

    # Plot the time-domain signal and its envelope
    plt.figure(figsize=(12, 5))
    plt.plot(Y[:, selected_channel], label="Original Signal")
    plt.plot(envelope, "--r", label="Envelope (Hilbert)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Channel {selected_channel}, Padding {pad}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------------
    # Next Steps: Hilbert PCA
    # ---------------------------
    U, s, VT = np.linalg.svd(Ya, full_matrices=False)
    U = U / U.std()
    expvar = s**2 / (U.shape[0] - 1)
    frac_expvar = expvar / expvar.sum()

    fig, ax = plt.subplots(1, 3, figsize=(12.2, 4), dpi=300)

    ax[0].plot(U.real[:, 0])
    # ax[0].plot(U.imag[:, 0])
    ax[0].plot(np.abs(U)[:, 0], color=".3")

    ax[1].plot(U[:, 1].real)
    # ax[1].plot(U[:, 1].imag)
    ax[1].plot(np.abs(U)[:, 1], color=".3")

    ref = sources["ar1"]
    ref = (ref - ref.mean()) / ref.std()
    ax[2].plot(U[:, 2].real)
    ax[2].plot(ref)
    # ax[2].plot(U[:, 2].imag)
    ax[2].plot(np.abs(U)[:, 2], color=".3")

    ax[0].set_title("{:.2f} %".format(frac_expvar[0]))
    ax[1].set_title("{:.2f} %".format(frac_expvar[1]))
    ax[2].set_title("{:.2f} %".format(frac_expvar[2]))
    plt.show()

# %%


# %%
signal = "trend"
pad = "exp_lin"
Ya = transform_hilbert(Y, ext=pad, alpha=4e-2)
U, s, VT = np.linalg.svd(Ya, full_matrices=False)
U = U / U.std(0)

fs = 100
N = U.shape[0]


x = sources[signal]
x_ = U[:, 1]
x = (x - x.mean()) / x.std()
x_ = (x_ - x_.mean()) / x_.std()


N = len(x)
fourier_true = np.fft.fft(x)
fourier_pca = np.fft.fft(x_)

# Compute the two-sided PSD
PSD_true = (1 / (fs * N)) * np.abs(fourier_true) ** 2
PSD_pca = (1 / (fs * N)) * np.abs(fourier_pca) ** 2

# Create frequency bins (two-sided)
freq = np.fft.fftfreq(N, d=1 / fs)

# Shift zero frequency component to the center (optional)
PSD_true = np.fft.fftshift(PSD_true)
PSD_pca = np.fft.fftshift(PSD_pca)
freq = np.fft.fftshift(freq)


plt.plot(x.real)
plt.plot(x_.real)
plt.show()

plt.plot(freq, PSD_true, lw=2)
plt.plot(freq, PSD_pca)
plt.xlim(-3, 3)
plt.show()

# %%
