import numpy as np
import matplotlib.pyplot as plt
from functions import postprocessing
from scipy.signal import find_peaks



def plot_cp(distances, parameters, window_size, time_start, time_stop, threshold, plot_prominences=False,
            simulate_data=True, weights=None):
    """
    Plots dissimilarity measure with ground-truth changepoints

    Args:
        distances: dissimilarity measures
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD
        time_start: first time stamp of plot
        time_stop: last time stamp of plot
        plot_prominences: True/False

    Returns:
        plot of dissimilarity measure with ground-truth changepoints

    """
    if simulate_data:
        breakpoints = postprocessing.parameters_to_cps(parameters, window_size)
    else:
        distances = np.concatenate((np.zeros((window_size,)), distances, np.zeros((window_size,))))
        if len(parameters) > len(distances):
            distances = np.pad(distances, (0, len(parameters) - len(distances) + 1))
        if weights is not None:
            weights = np.concatenate((np.zeros((2, window_size)), weights, np.zeros((2, window_size))), axis=1)
        breakpoints = parameters.copy()
        breakpoints.append(0)
        np.append(breakpoints, 0)

    t = range(len(distances))

    x = t
    z = distances
    y = breakpoints  # [:,0]
    cps = [idx for idx in range(len(y)) if y[idx] > 0]
    peaks = find_peaks(distances)[0]
    peaks_prom_all = np.array(postprocessing.new_peak_prominences(distances)[0])

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(x, z, color="black")
    hit_list = []
    for cp in cps:
        if peaks.size == 0:
            break
        nearst_peak = peaks[np.abs(peaks - cp).argmin()]
        if np.abs(nearst_peak - cp) <= threshold:
            peaks = np.delete(peaks, np.where(peaks == nearst_peak))
            hit_list.append(nearst_peak)

    if plot_prominences:
        ax.plot(x, peaks_prom_all, color="blue")

    ax.set_xlim(time_start, time_stop)
    ax.set_ylim(0, 1.5 * max(z))
    plt.xlabel("time")
    plt.ylabel("dissimilarity")
    ax.plot(peaks, distances[peaks], 'ko', label='false positives')
    reset_flags = {"A&S": True, "B": True}
    if weights is None:
        ax.plot(hit_list, distances[hit_list], 'go', label='accurately detected')
        ax.legend()
    else:
        for idx, item in enumerate(hit_list):
            # weight = tuple(weights[:, item])
            # ax.plot(item, distances[item], marker='o', color=weight)
            cls = np.argmax(weights, axis=0)[item]
            if cls == 0:
                if reset_flags["A&S"]:
                    ax.plot(item, distances[item], 'go', label='changes detected as A or S type')
                    reset_flags["A&S"] = False
                else:
                    ax.plot(item, distances[item], 'go')
            elif cls == 1:
                if reset_flags["B"]:
                    ax.plot(item, distances[item], 'ro', label='changes detected as B type')
                    reset_flags["B"] = False
                else:
                    ax.plot(item, distances[item], 'ro')
        ax.legend()
    height_line = 1

    ax.fill_between(x[0:len(y)], 0, height_line, where=y > 0.0001 * np.ones_like(y),
                    color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.fill_between(x[0:len(y)], 0, height_line, where=y > 0.25 * np.ones_like(y),
                    color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.fill_between(x[0:len(y)], 0, height_line, where=y > 0.5 * np.ones_like(y),
                    color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.fill_between(x[0:len(y)], 0, height_line, where=y > 0.75 * np.ones_like(y),
                    color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.fill_between(x[0:len(y)], 0, height_line, where=y > 0.9 * np.ones_like(y),
                    color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    plt.show()


def cal_f1(bps, distances, tol_dist, peaks=None):
    if peaks is not None:
        peaks = peaks
    else:
        peaks = find_peaks(distances)[0]
    hit_list = []  # TP
    miss_list = []  # FN
    for bp in bps:
        if peaks.size == 0:
            break
        nearst_peak = peaks[np.abs(peaks - bp).argmin()]
        if np.abs(nearst_peak - bp) <= tol_dist:
            peaks = np.delete(peaks, np.where(peaks == nearst_peak))
            hit_list.append(nearst_peak)
        else:
            miss_list.append(bp)
    n_tp = len(hit_list)
    n_fn = len(miss_list)
    n_fp = peaks.shape[0]
    precision = n_tp / (n_tp + n_fp + 1e-8)
    recall = n_tp / (n_tp + n_fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return precision, recall, f1


def print_f1(distances, tol_distance, parameters, window_size, simulate_data=True, peaks=None):
    """
    Calculation of AUC for toleration distances in range(TD_start, TD_stop, TD_step) + plot of corresponding ROC curves

    Args:
        distances: dissimilarity measures
        tol_distances: list of different toleration distances
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD

    Returns:
        list of AUCs for every toleration distance
    """
    if peaks is not None:
        if simulate_data:
            breakpoints = postprocessing.parameters_to_cps(parameters, window_size)
        else:
            breakpoints = parameters.copy()
        list_of_lists = postprocessing.cp_to_timestamps(breakpoints, 0, np.size(breakpoints))
        precision, recall, f1 = cal_f1(list_of_lists, distances, tol_distance, peaks)
    else:
        if simulate_data:
            breakpoints = postprocessing.parameters_to_cps(parameters, window_size)
        else:
            distances = np.concatenate((np.zeros((window_size,)), distances, np.zeros((window_size,))))
            breakpoints = parameters.copy()
        list_of_lists = postprocessing.cp_to_timestamps(breakpoints, 0, np.size(breakpoints))
        precision, recall, f1 = cal_f1(list_of_lists, distances, tol_distance)
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("f1-score:" + str(f1))

    return precision, recall, f1
