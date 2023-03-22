from functions import utils, data_loader
from functions import evaluation, postprocessing
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# --------------------------- #
# SET PARAMETERS
seed = np.random.randint(0, 1e8, size=1).tolist()[0]
generate_data = False
architecture = 'TIRE_cnn_new'
dataset = "ar5"                   # mean0-9, var0-9, gauss0-9, ar0-9, hasc, bee_dance0-5, ..., if generate_data = False
enable_eval_plot = True
enable_model_summary = False
domain = "both"
nfft = 30   # number of points for DFT
norm_mode = "timeseries"   # for calculation of DFT, should the timeseries have mean zero or each window?

# --------------------------- #
# BEGINNING OF CODE
# load data and model
window_size, threshold = utils.set_windowsize_and_threshold(dataset)
time_series, windows_TD, windows_FD, parameters = data_loader.data_parse(nfft, norm_mode, generate_data, dataset, window_size)
if len(time_series.shape) == 1:
    time_series = np.expand_dims(time_series, axis=-1)
    windows_TD = np.expand_dims(windows_TD, axis=-1)
    windows_FD = np.expand_dims(windows_FD, axis=-1)

network = utils.load_architecture(architecture)
path = os.path.join(os.path.abspath(os.getcwd()), "results")

gts = []
change_points = []
for idx in range(np.shape(parameters)[0] - 1):
    pre = parameters[idx]
    suc = parameters[idx + 1]
    if pre != suc:
        change_points.append(1)
        gts.append(idx + 1)
    else:
        change_points.append(0)

#---------------------------#
##TRAIN THE AUTOENCODERS
shared_features_TD = network.train_model(windows_TD, enable_summary=True, window_size=window_size, seed=seed)
shared_features_FD = network.train_model(windows_FD, enable_summary=True, window_size=window_size, seed=seed)
beta = np.quantile(postprocessing.distance(shared_features_TD, window_size), 0.95)
alpha = np.quantile(postprocessing.distance(shared_features_FD, window_size), 0.95)
encoded_windows_both = np.concatenate((shared_features_TD * alpha, shared_features_FD * beta), axis=1)

dissimilarities1 = evaluation.smoothened_dissimilarity_measures(shared_features_TD, shared_features_FD, "TD", window_size)
dissimilarities2 = evaluation.smoothened_dissimilarity_measures(shared_features_TD, shared_features_FD, "FD", window_size)
dissimilarities3 = evaluation.smoothened_dissimilarity_measures(shared_features_TD, shared_features_FD, "both", window_size)

print("Time Domain:")
evaluation.show_result(generate_data, window_size, dissimilarities1, parameters, threshold, enable_eval_plot)
print("Frequency Domain:")
evaluation.show_result(generate_data, window_size, dissimilarities2, parameters, threshold, enable_eval_plot)
print("Both Domain:")
evaluation.show_result(generate_data, window_size, dissimilarities3, parameters, threshold, enable_eval_plot)
