import numpy as np
import pywt
import warnings


def load_architecture(architecture_name):
    """Loads the network definition from the architectures folder

    :param architecture_name: String with the name of the network
    :return: module with the architecture
    """
    return getattr(__import__('architectures.%s' % architecture_name), '%s' % architecture_name)


def set_windowsize_and_threshold(dataset):
    if (("AR" in dataset) and not("VAR" in dataset)) \
            or (("ar" in dataset) and not("var" in dataset)) or ("MC" in dataset) or ("feature" in dataset):
        windowsize = 40
        threshold = 40
    elif "bee" in dataset:
        windowsize = 16
        threshold = 16
    elif "hasc" in dataset:
        windowsize = 300
        threshold = 300
    elif "UCI" in dataset:
        windowsize = 8
        threshold = 8
    elif "change" in dataset:
        windowsize = 40
        threshold = 40
    elif "EDP" in dataset:
        windowsize = 30
        threshold = 30
    elif "80711" in dataset:
        windowsize = 30
        threshold = 30
    elif "baby" in dataset:
        windowsize = 20
        threshold = 20
    elif "well" in dataset:
        windowsize = 100
        threshold = 100
    elif ("cycle" in dataset) or ("walk" in dataset) or ("run" in dataset) or ("stand" in dataset):
        windowsize = 300
        threshold = 300
    else:
        for keyword in ["mean", "MEAN", "var", "VAR", "gauss", "GAUSS", "ar"]:
            if keyword in dataset:
                windowsize = 40
                threshold = 40
    return windowsize, threshold


def calc_fft(windows, nfft=30, norm_mode="timeseries"):
    """
    Calculates the DFT for each window and transforms its length

    Args:
        windows: time series windows
        nfft: number of points used for the calculation of the DFT
        norm_mode: ensure that the timeseries / each window has zero mean

    Returns:
        frequency domain windows, each window having size nfft//2 (+1 for timeseries normalization)
    """
    mean_per_segment = np.mean(windows, axis=-1)
    mean_all = np.nanmean(mean_per_segment, axis=0)

    if norm_mode == "window":
        windows = windows - mean_per_segment[:, None]
        windows_fft = abs(np.fft.fft(windows, nfft))[..., 1:nfft // 2 + 1]
    elif norm_mode == "timeseries":
        windows = windows - mean_all
        windows_fft = abs(np.fft.fft(windows, nfft))[..., :nfft // 2 + 1]

    fft_max = np.amax(windows_fft)
    fft_min = np.amin(windows_fft)
    windows_fft = 2 * (windows_fft - fft_min) / (fft_max - fft_min) - 1

    return windows_fft


def norm_windows(windows):
    fft_max = np.amax(windows)
    fft_min = np.amin(windows)
    windows_normed = 2 * (windows - fft_min) / (fft_max - fft_min) - 1
    return windows_normed


def minmaxscale(data, a, b):
    """
    Scales data to the interval [a,b]
    """
    data_min = np.amin(data)
    data_max = np.amax(data)
    return (b-a)*(data-data_min)/(data_max-data_min)+a


def overlap_test(list1, list2, threshold):
    for element in list1:
        if list2.size == 0:
            break
        nearst_element = list2[np.abs(list2 - element).argmin()]
        if np.abs(nearst_element - element) <= threshold:
            list1 = np.delete(list1, np.where(list1 == element))
    return list1


def fft_denoiser(x, n_components, to_real=True):
    """Fast fourier transform denoiser.

    Denoises data using the fast fourier transform.

    Parameters
    ----------
    x : numpy.array
        The data to denoise.
    n_components : int
        The value above which the coefficients will be kept.
    to_real : bool, optional, default: True
        Whether to remove the complex part (True) or not (False)

    Returns
    -------
    clean_data : numpy.array
        The denoised data.

    References
    ----------
    .. [1] Steve Brunton - Denoising Data with FFT[Python]
       https://www.youtube.com/watch?v=s2K1JfNR7Sc&ab_channel=SteveBrunton

    """
    n = len(x)

    # compute the fft
    fft = np.fft.fft(x, n)

    # compute power spectrum density
    # squared magnitud of each fft coefficient
    PSD = (fft * np.conj(fft) / n).real
    PSD = PSD / np.max(PSD)
    # keep high frequencies
    total_sum = np.sum(PSD)
    _mask = []
    sum = 0
    for i in PSD:
        sum = sum + i
        if sum >= n_components * total_sum:
            _mask.append(0)
        else:
            _mask.append(1)
    # _mask = PSD > n_components
    fft = _mask * fft

    # inverse fourier transform
    clean_data = np.fft.ifft(fft)

    if to_real:
        clean_data = clean_data.real

    return clean_data


def wavelet_denoising(data):
    """ Add positional encoding to extracted features.
    Parameters
    ----------
    data: numpy.array
        the input time sequence to be denoised.

    Returns
    -------
    meta: numpy.array
        denoised time sequence.
    """
    db4 = pywt.Wavelet('db4')
    if data is not None:
        # decomposition
        coeffs = pywt.wavedec(data, db4)
        # set high freq coeff to zero
        for idx, coeff in enumerate(coeffs):
            if idx > len(coeffs)//3:
                coeffs[idx] *= 0
        # reconstruction
        meta = pywt.waverec(coeffs, db4)
        return meta[:len(data)]


# positional encoding
def positional_encoding(encoded_windows_both, feature_dimension, merge="add"):
    """ Add positional encoding to extracted features.
    Parameters
    ----------
    encoded_windows_both: numpy.array
        the extracted features.
    feature_dimension: int
        the dimension of positional features.
    merge: str, optional, default: "add"
        the operation to merge the positional features and the extracted features.

    Returns
    -------
    positionalized_windows: numpy.array
        The features after positional encoding.
    """
    max_len = len(encoded_windows_both)  # length of features
    d_model = feature_dimension   # dimension of positional vector
    pe = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(d_model // 2):
            pe[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / d_model))
            pe[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / d_model))
    pe = minmaxscale(pe, encoded_windows_both.min(), encoded_windows_both.max())
    if merge == "concat":
        positionalized_windows = np.concatenate((pe, encoded_windows_both), axis=-1)
    elif merge == "add":
        positionalized_windows = (pe + encoded_windows_both).astype(np.float32)
    else:
        warnings.warn("Invalid merge method! Automatically use \"add\" instead.")
        positionalized_windows = (pe + encoded_windows_both).astype(np.float32)

    return positionalized_windows, pe


def edge_detection(img):
    # define the vertical filter
    vertical_filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    # define the horizontal filter
    horizontal_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    # get the dimensions of the image
    n, m = img.shape

    # initialize the edges image
    edges_img = img.copy()

    # loop over all pixels in the image
    for row in range(3, n - 2):
        for col in range(3, m - 2):
            # create little local 3x3 box
            local_pixels = img[row - 1:row + 2, col - 1:col + 2]

            # apply the vertical filter
            vertical_transformed_pixels = vertical_filter * local_pixels
            # remap the vertical score
            vertical_score = vertical_transformed_pixels.sum() / 4

            # apply the horizontal filter
            horizontal_transformed_pixels = horizontal_filter * local_pixels
            # remap the horizontal score
            horizontal_score = horizontal_transformed_pixels.sum() / 4

            # combine the horizontal and vertical scores into a total edge score
            edge_score = (vertical_score ** 2 + horizontal_score ** 2) ** .5

            # insert this edge score into the edges image
            edges_img[row, col] = edge_score

    # remap the values in the 0-1 range in case they went out of bounds
    edges_img = edges_img / edges_img.max()
    return edges_img


def array_to_image(matrix):
    _min = np.amin(matrix)
    _max = np.amax(matrix)
    img_norm = (matrix - _min) * 255.0 / (_max - _min)
    img_norm = np.uint8(img_norm)
    return img_norm