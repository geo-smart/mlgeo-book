"""Synthetic seismograms and spectrum-matched noise.

Stands in for: short windows of 100 Hz three-component ground motion around a
local earthquake, in the spirit of template-matching and deep-learning
detection datasets (e.g. ConvNetQuake-style earthquake/noise classification).

Breaks down: the source is a simple damped-oscillation wavelet, not a rupture
model; there is one P-like and one S-like arrival with fixed apparent
velocities; path effects reduce to attenuation and geometric spreading. Real
seismograms carry site response, scattering coda, and instrument response.
"""

import numpy as np


def _wavelet(t, f0, tau):
    """Damped-cosine source wavelet with dominant frequency f0 (Hz)."""
    w = np.zeros_like(t)
    pos = t >= 0
    w[pos] = np.cos(2 * np.pi * f0 * t[pos]) * np.exp(-t[pos] / tau)
    return w


def synthetic_seismogram(
    duration_s=30.0,
    fs=100.0,
    magnitude=2.0,
    distance_km=20.0,
    f0=5.0,
    snr=5.0,
    noise_color=1.0,
    seed=0,
):
    """One vertical-component synthetic earthquake record in arbitrary velocity units.

    Amplitude scales as 10**magnitude / distance (geometric spreading) with
    exponential attenuation; P arrives at distance/6.0 km/s, S at
    distance/3.5 km/s with twice the P amplitude and lower dominant frequency.
    ``snr`` sets the ratio of peak signal amplitude to noise standard deviation
    — the detection-floor knob used in Chapter 4.3.

    Returns (t, trace, meta) where meta records arrival times and true SNR.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    tp = 5.0 + distance_km / 6.0
    ts = 5.0 + distance_km / 3.5
    amp = 10.0 ** (magnitude - 2.0) / max(distance_km, 1.0) * np.exp(-distance_km / 100.0)

    signal = amp * _wavelet(t - tp, f0, 0.5) + 2.0 * amp * _wavelet(t - ts, 0.6 * f0, 1.2)

    freqs = np.fft.rfftfreq(n, d=1 / fs)
    namp = np.ones_like(freqs)
    namp[1:] = freqs[1:] ** (-noise_color / 2.0)
    namp[0] = 0.0
    spec = namp * np.exp(1j * rng.uniform(0, 2 * np.pi, len(freqs)))
    noise = np.fft.irfft(spec, n=n)
    noise = noise / noise.std()

    peak = np.abs(signal).max()
    noise = noise * (peak / snr if peak > 0 else 1.0)
    meta = {"t_p": tp, "t_s": ts, "peak_amplitude": peak, "snr": snr}
    return t, signal + noise, meta


def seismogram_dataset(n_events=500, n_noise=500, fs=100.0, duration_s=30.0, seed=0):
    """Labeled 2-class dataset of event and noise windows for detection lessons.

    Events draw magnitude ~ U(1, 4), distance ~ U(5, 80) km, SNR ~ logU(0.5, 20).
    Returns (X, y, meta_list): X has shape (n, duration*fs), y is 1 for event.
    """
    rng = np.random.default_rng(seed)
    n_samp = int(duration_s * fs)
    X = np.zeros((n_events + n_noise, n_samp))
    y = np.zeros(n_events + n_noise, dtype=int)
    metas = []
    for i in range(n_events):
        _, tr, meta = synthetic_seismogram(
            duration_s,
            fs,
            magnitude=rng.uniform(1, 4),
            distance_km=rng.uniform(5, 80),
            snr=float(np.exp(rng.uniform(np.log(0.5), np.log(20)))),
            seed=rng.integers(2**31),
        )
        X[i] = tr / np.abs(tr).max()
        y[i] = 1
        metas.append(meta)
    for i in range(n_noise):
        _, tr, meta = synthetic_seismogram(
            duration_s, fs, magnitude=-10, snr=1e-12, seed=rng.integers(2**31)
        )
        X[n_events + i] = tr / np.abs(tr).max()
        metas.append({"noise": True})
    return X, y, metas


def spectrum_matched_noise(reference, n=None, seed=0):
    """Random-phase noise whose amplitude spectrum matches ``reference``.

    This is the physics-informed noise idea from Chapter 2.10: keep the
    amplitude spectrum of a real noise record, randomize the phase. Uses
    rfft/irfft so Hermitian symmetry — and therefore a real-valued output —
    is guaranteed by construction (the 2024 edition hand-rolled this and got
    the conjugate symmetry wrong).
    """
    rng = np.random.default_rng(seed)
    reference = np.asarray(reference, dtype=float)
    if n is None:
        n = len(reference)
    amp = np.abs(np.fft.rfft(reference, n=n))
    phases = rng.uniform(0, 2 * np.pi, len(amp))
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0  # Nyquist bin must stay real for a real-valued output
    spec = amp * np.exp(1j * phases)
    return np.fft.irfft(spec, n=n)
