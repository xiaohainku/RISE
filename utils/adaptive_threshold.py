from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import numpy as np


def KDE(kde_data, default_thr=None, minimal_pos=None, min_thr=None, max_thr=None):
    kde = gaussian_kde(kde_data)
    x_vals = np.linspace(min(kde_data), max(kde_data), 10000)
    kde_vals = kde(x_vals)

    inverted_kde = -kde_vals
    peaks, _ = find_peaks(inverted_kde)
    minima_x = x_vals[peaks]
    minima_x = [x for x in minima_x if min_thr < x < max_thr]

    thr = minima_x[minimal_pos] if len(minima_x) > 0 else default_thr
    return thr


def KDE_RISE(kde_data, minimal_pos=0, default_thr=0.2):
    kde = gaussian_kde(kde_data)
    x_vals = np.linspace(min(kde_data), max(kde_data), 10000)
    kde_vals = kde(x_vals)

    peaks, _ = find_peaks(kde_vals)
    minima_x = x_vals[peaks]

    thr = minima_x[minimal_pos] if len(minima_x) > 0 else default_thr
    return thr


def HIST_RISE(data):
    n, bins = np.histogram(data, bins=50)
    max_freq_index = np.argmax(n)
    max_freq_bin = (bins[max_freq_index] + bins[max_freq_index + 1]) / 2
    return max_freq_bin


def KDE_Entropy(kde_data):
    kde = gaussian_kde(kde_data)
    x_vals = np.linspace(min(kde_data), max(kde_data), 10000)
    kde_vals = kde(x_vals)

    inverted_kde = -kde_vals
    peaks, _ = find_peaks(inverted_kde)
    minima_x = x_vals[peaks]
    thr = minima_x[-1] if len(minima_x) > 0 else 0.2

    p = kde.evaluate(x_vals)
    H = -np.trapz(p * np.log(p + 1e-12), x_vals)
    return thr, H


from scipy.stats import wasserstein_distance


def emd_adjustment(data, mask, iterations=10):
    for _ in range(iterations):
        group1 = data[mask == 1]
        group2 = data[mask == 0]

        if len(group1) == 0 or len(group2) == 0:
            break

        centroid1 = np.mean(group1)
        centroid2 = np.mean(group2)

        new_mask = np.zeros_like(mask)
        for idx, point in enumerate(data):
            distance_to_c1 = wasserstein_distance([point], [centroid1])
            distance_to_c2 = wasserstein_distance([point], [centroid2])

            new_mask[idx] = 1 if distance_to_c1 < distance_to_c2 else 0

        if np.array_equal(new_mask, mask):
            break

        mask = new_mask

    return mask
