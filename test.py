import numpy as np
import matplotlib.pyplot as plt
import time as tm

# pip install pyinform
from pyinform.blockentropy import block_entropy as pyinform_block_entropy


# ============================================================
# 1) UTILITIES
# ============================================================

def shannon_entropy_from_counts(counts, base=2):
    """
    Shannon entropy from counts.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return np.nan

    probs = counts[counts > 0] / total
    return -np.sum(probs * np.log(probs) / np.log(base))


def discretize_signal(x, n_bins=10):
    """
    Discretize a continuous 1D signal into integer symbols 0, 1, ..., n_bins-1.
    """
    x = np.asarray(x).ravel()

    if x.size == 0:
        return np.array([], dtype=int)

    xmin = np.min(x)
    xmax = np.max(x)

    # constant window
    if xmin == xmax:
        return np.zeros_like(x, dtype=int)

    edges = np.linspace(xmin, xmax, n_bins + 1)

    # output in {0, ..., n_bins-1}
    symbols = np.digitize(x, edges[1:-1], right=False)
    return symbols.astype(int)


# ============================================================
# 2) CUSTOM BLOCK ENTROPY (k=2)
# ============================================================

def custom_block_entropy_k2_from_symbols(symbols, n_bins, base=2):
    """
    Custom block entropy of order k=2 from an already discretized symbol sequence.

    symbols must be integers in {0, ..., n_bins-1}.
    """
    symbols = np.asarray(symbols).ravel()

    if len(symbols) < 2:
        return np.nan

    # encode each pair (s_t, s_{t+1}) as one integer
    pair_ids = symbols[:-1] * n_bins + symbols[1:]

    counts = np.bincount(pair_ids, minlength=n_bins * n_bins)
    return shannon_entropy_from_counts(counts, base=base)


# ============================================================
# 3) SLIDING WINDOW COMPARISON
# ============================================================

def sliding_window_block_entropy_comparison(
    signal,
    window_size,
    step=1,
    n_bins=10,
    k=2,
    base=2
):
    """
    Compare:
      - custom block entropy (k=2)
      - PyInform block entropy (k=2)

    on sliding windows of a 1D signal.

    Returns
    -------
    centers : np.ndarray
    custom_vals : np.ndarray
    pyinform_vals : np.ndarray
    custom_times : np.ndarray
    pyinform_times : np.ndarray
    """
    if k != 2:
        raise ValueError("This implementation compares only k=2 for the custom method.")

    signal = np.asarray(signal).ravel()

    if window_size < k:
        raise ValueError("window_size must be >= k.")

    starts = np.arange(0, len(signal) - window_size + 1, step)
    centers = starts + window_size // 2

    custom_vals = []
    pyinform_vals = []
    custom_times = []
    pyinform_times = []

    for s in starts:
        w = signal[s:s + window_size]

        # Fair comparison: same discretization for both methods
        symbols = discretize_signal(w, n_bins=n_bins)

        t0 = tm.perf_counter()
        h_custom = custom_block_entropy_k2_from_symbols(symbols, n_bins=n_bins, base=base)
        t1 = tm.perf_counter()

        t2 = tm.perf_counter()
        h_pyinform = pyinform_block_entropy(symbols.tolist(), k)
        t3 = tm.perf_counter()

        custom_vals.append(h_custom)
        pyinform_vals.append(h_pyinform)
        custom_times.append(t1 - t0)
        pyinform_times.append(t3 - t2)

    return (
        np.asarray(centers),
        np.asarray(custom_vals),
        np.asarray(pyinform_vals),
        np.asarray(custom_times),
        np.asarray(pyinform_times),
    )


# ============================================================
# 4) VISUALIZATION
# ============================================================

def plot_block_entropy_comparison(
    signal,
    centers,
    custom_vals,
    pyinform_vals,
    custom_times,
    pyinform_times
):
    """
    Visualize:
      - raw signal
      - both entropy curves
      - scatter/correlation
      - absolute difference
      - timing comparison
    """
    valid = np.isfinite(custom_vals) & np.isfinite(pyinform_vals)

    if valid.sum() > 1:
        corr = np.corrcoef(custom_vals[valid], pyinform_vals[valid])[0, 1]
    else:
        corr = np.nan

    diff = custom_vals - pyinform_vals
    abs_diff = np.abs(diff)

    fig = plt.figure(figsize=(15, 12))

    # 1) raw signal
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(signal, linewidth=1)
    ax1.set_title("Raw signal")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Amplitude")

    # 2) entropy series
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(centers, custom_vals, label="Custom block entropy (k=2)", linewidth=2)
    ax2.plot(centers, pyinform_vals, label="PyInform block entropy (k=2)", linewidth=2)
    ax2.set_title(f"Sliding-window block entropy\nCorrelation = {corr:.6f}")
    ax2.set_xlabel("Window center")
    ax2.set_ylabel("Entropy")
    ax2.legend()

    # 3) scatter
    ax3 = plt.subplot(3, 2, 3)
    ax3.scatter(custom_vals[valid], pyinform_vals[valid], alpha=0.6)
    ax3.set_title("PyInform vs custom")
    ax3.set_xlabel("Custom block entropy")
    ax3.set_ylabel("PyInform block entropy")

    # 4) difference
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(centers, diff, linewidth=1.5)
    ax4.axhline(0, linestyle="--", linewidth=1)
    ax4.set_title("Difference: custom - PyInform")
    ax4.set_xlabel("Window center")
    ax4.set_ylabel("Difference")

    # 5) absolute difference
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(centers, abs_diff, linewidth=1.5)
    ax5.set_title("Absolute difference")
    ax5.set_xlabel("Window center")
    ax5.set_ylabel("|custom - PyInform|")

    # 6) timing
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(centers, custom_times * 1e3, label="Custom time", linewidth=2)
    ax6.plot(centers, pyinform_times * 1e3, label="PyInform time", linewidth=2)
    ax6.set_title("Per-window computation time")
    ax6.set_xlabel("Window center")
    ax6.set_ylabel("Time (ms)")
    ax6.legend()

    plt.tight_layout()
    plt.show()

    print("===== RESULT SUMMARY =====")
    print(f"Correlation(custom, pyinform)      : {corr:.8f}")
    print(f"Mean absolute difference           : {np.nanmean(abs_diff):.8e}")
    print(f"Max absolute difference            : {np.nanmax(abs_diff):.8e}")

    print("\n===== TIMING SUMMARY =====")
    print(f"Average custom time per window     : {np.mean(custom_times)*1e3:.6f} ms")
    print(f"Average PyInform time per window   : {np.mean(pyinform_times)*1e3:.6f} ms")
    print(f"Median custom time per window      : {np.median(custom_times)*1e3:.6f} ms")
    print(f"Median PyInform time per window    : {np.median(pyinform_times)*1e3:.6f} ms")
    print(f"Total custom time                  : {np.sum(custom_times):.6f} s")
    print(f"Total PyInform time                : {np.sum(pyinform_times):.6f} s")

    ratio = np.mean(custom_times) / np.mean(pyinform_times) if np.mean(pyinform_times) > 0 else np.nan
    print(f"Mean time ratio (custom / pyinform): {ratio:.4f}")


# ============================================================
# 5) EXAMPLE SIGNAL
# ============================================================

def make_example_signal(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    sig = np.zeros(n, dtype=float)

    sig[:1500] = np.sin(2 * np.pi * t[:1500] / 40) + 0.1 * rng.standard_normal(1500)
    sig[1500:3000] = 0.5 * np.sin(2 * np.pi * t[1500:3000] / 17) + 0.8 * rng.standard_normal(1500)
    sig[3000:] = (
        0.7 * np.sin(2 * np.pi * t[3000:] / 60)
        + 0.3 * np.sin(2 * np.pi * t[3000:] / 11)
        + 0.25 * rng.standard_normal(n - 3000)
    )

    return sig


# ============================================================
# 6) MAIN
# ============================================================
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from Functions.detect_artifacts import find_artifacts
from Functions.WaveletQuantileNormalization import WQN_3
from Prediction.extract_data import get_data, get_part

data = sc.io.loadmat('Data/rec_20230629_092502.mat')
y = data['record'][:,0]
fs = 128

if __name__ == "__main__":
    # Replace with your own 1D signal if needed
    signal = y

    window_size = 90 * fs
    step = 45 * fs
    n_bins = 10
    k = 2

    centers, custom_vals, pyinform_vals, custom_times, pyinform_times = (
        sliding_window_block_entropy_comparison(
            signal=signal,
            window_size=window_size,
            step=step,
            n_bins=n_bins,
            k=k,
            base=2
        )
    )

    plot_block_entropy_comparison(
        signal,
        centers,
        custom_vals,
        pyinform_vals,
        custom_times,
        pyinform_times
    )