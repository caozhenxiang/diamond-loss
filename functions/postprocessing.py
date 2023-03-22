import warnings
import numpy as np
from scipy.signal import peak_prominences
import faiss


def distance(data, window_size):
    """
    Calculates distance (dissimilarity measure) between features

    Args:
        data: array of of learned features of size (nr. of windows) x (number of shared features)
        window_size: window size used for CPD

    Returns:
        Array of dissimilarities of size ((nr. of windows)-stride)
    """
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=-1)
    nr_windows, nr_c = np.shape(data)

    index_1 = range(window_size, nr_windows, 1)
    index_2 = range(0, nr_windows - window_size, 1)
    return np.sqrt(np.sum(np.square(data[index_1] - data[index_2]), 1))


def new_peak_prominences(distances):
    """
    Adapted calculation of prominence of peaks, based on the original scipy code

    Args:
        distances: dissimarity scores
    Returns:
        prominence scores
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        all_peak_prom = peak_prominences(distances, range(len(distances)))
    return all_peak_prom


def parameters_to_cps(parameters, window_size):
    """
    Preparation for plotting ground-truth change points

    Args:
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD

    Returns:
        Array of which entry is non-zero in the presence of a change point. Higher values correspond to larger parameter changes.
    """

    length_ts = np.size(parameters ,0)

    index1 = range(window_size -1 ,length_ts -window_size ,1)   # selects parameter at LAST time stamp of window
    index2 = range(window_size ,length_ts -window_size +1 ,1)   # selects parameter at FIRST time stamp of next window
    diff_parameters = np.sqrt(np.sum(np.square(parameters[index1 ] -parameters[index2]) ,1))

    max_diff = np.max(diff_parameters)

    return diff_parameters /max_diff


def cp_to_timestamps(changepoints, tolerance, length_ts):
    """
    Extracts time stamps of change points

    Args:
        changepoints:
        tolerance:
        length_ts: length of original time series

    Returns:
        list where each entry is a list with the windows affected by a change point
    """

    locations_cp = [idx for idx, val in enumerate(changepoints) if val > 0.0]

    output = []
    while len(locations_cp) > 0:
        k = 0
        output.append \
            (list(range(max(locations_cp[0] - tolerance, 0), min(locations_cp[k] + 1 +tolerance, length_ts), 1)))
        del locations_cp[:k+1]

    return output


def matched_filter(signal, window_size):
    """
    Matched filter for dissimilarity measure smoothing (and zero-delay weighted moving average filter for shared feature smoothing)

    Args:
        signal: input signal
        window_size: window size used for CPD
    Returns:
        filtered signal
    """
    mask = np.ones((2 * window_size + 1,))
    for i in range(window_size):
        mask[i] = i / (window_size ** 2)
        mask[-(i + 1)] = i / (window_size ** 2)
    mask[window_size] = window_size / (window_size ** 2)

    signal_out = np.zeros(np.shape(signal))

    if len(np.shape(signal)) > 1:  # for both TD and FD
        for i in range(np.shape(signal)[1]):
            signal_extended = np.concatenate((signal[0, i] * np.ones(window_size), signal[:, i], signal[-1, i] *
                                              np.ones(window_size)))
            signal_out[:, i] = np.convolve(signal_extended, mask, 'valid')
    else:
        signal = np.concatenate((signal[0] * np.ones(window_size), signal, signal[-1] * np.ones(window_size)))
        signal_out = np.convolve(signal, mask, 'valid')

    return signal_out


def neighbors_seeking(encoded_windows_both, feature_dimension, nr_NN):
    """ Search for the nearst neighbours of each extracted features
    Parameters
    ----------
    encoded_windows_both: numpy.array
        the extracted features.
    feature_dimension: int
        the dimension of positional features.
    nr_NN: int
        the number of nearst neighbours to be found for each feature.

    Returns
    -------
    NN: numpy.array
        the location (index) of nearst neighbours for each feature.
    """
    index = faiss.IndexFlatL2(feature_dimension)
    encoded_windows_both = encoded_windows_both.astype(np.float32)
    index.add(encoded_windows_both)
    D, I = index.search(encoded_windows_both, nr_NN + 1)
    idx_mat = np.tile(np.expand_dims(np.arange(len(I)), axis=-1), (1, nr_NN + 1))
    NN_idx = list(map(lambda x: np.argwhere(x != 0)[0], [x for x in (I - idx_mat)]))
    NN = np.array([I[i, NN_idx[i]] for i in np.arange(len(NN_idx))])
    return NN


def get_arc_curve(NN):
    """ Compute the Arc Curve based on the index profile
    Parameters
    ----------
    NN: numpy.array
        the location (index) of nearst neighbours for each feature.

    Returns
    -------
    AC: numpy.array
        the number of linkages over each time sample.
    """
    n = len(NN)
    AC = np.zeros(n)
    nnmark = np.zeros(n)
    for i in np.arange(n):
        j = NN[i]
        nnmark[np.minimum(i, j)] = nnmark[np.minimum(i, j)] + 1
        nnmark[np.maximum(i, j)] = nnmark[np.maximum(i, j)] - 1
    numArcs = 0
    for i in np.arange(n):
        numArcs = numArcs + nnmark[i]
        AC[i] = numArcs
    return AC


def get_ideal_arc_curve(NN):
    """ Compute the ideal Arc Curve
    Parameters
    ----------
    NN: numpy.array
        the location (index) of nearst neighbours for each feature.

    Returns
    -------
    IAC: numpy.array
        the ideal Arc Curve for correction.
    """
    c1 = len(NN)/2
    c0 = -1/c1
    IAC = np.array(list(map(lambda x: c0*(x-c1)**2+c1, range(len(NN)))))
    return IAC

def get_corrected_arc_curve(AC, IAC, window_size):
    CAC = np.minimum(AC/IAC, 1)
    CAC[0:window_size] = np.max(CAC)
    CAC[-window_size:] = np.max(CAC)
    return CAC

