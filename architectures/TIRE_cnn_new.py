import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
if os.environ["CUDA_VISIBLE_DEVICES"] == "0":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Layer, Conv1D, Conv1DTranspose, Flatten, Reshape, PReLU, LeakyReLU
from tensorflow.keras.models import Model
import numpy as np

def create_parallel_AEs(X, enable_summary, seed):
    tf.keras.utils.set_random_seed(seed)
    initializer_cnn = tf.keras.initializers.GlorotUniform()

    # encoder
    input = Input(shape=(X.shape[1], X.shape[2], X.shape[3]), name="data")
    conv_S1 = input[:, 0, :, :]
    conv_S2 = input[:, 1, :, :]

    conv_S_L1 = Conv1D(filters=16, kernel_size=9, padding="same", kernel_initializer=initializer_cnn, strides=2,
                       activation=None, name="conv_S_L1")
    conv_S_L2 = Conv1D(filters=4, kernel_size=9, padding="same", kernel_initializer=initializer_cnn, strides=2,
                       activation="tanh", name="conv_S_L2")
    deconv_S_L2 = Conv1DTranspose(filters=16, kernel_size=9, padding="same", kernel_initializer=initializer_cnn,
                                  strides=2, activation=None, name="deconv_S_L1")
    deconv_S_L3 = Conv1DTranspose(filters=X.shape[3], kernel_size=9, padding="same", kernel_initializer=initializer_cnn,
                                  strides=2, activation="tanh", name="deconv_S_L2")
    prelu_enc = LeakyReLU()
    prelu_dec2 = LeakyReLU()

    conv_S1 = conv_S_L1(conv_S1)
    conv_S1 = prelu_enc(conv_S1)
    conv_S1 = conv_S_L2(conv_S1)

    conv_S2 = conv_S_L1(conv_S2)
    conv_S2 = prelu_enc(conv_S2)
    conv_S2 = conv_S_L2(conv_S2)

    shared_s1 = conv_S1[:, :, :-2]
    unshared_s1 = conv_S1[:, :, -2:]
    shared_s2 = conv_S2[:, :, :-2]
    unshared_s2 = conv_S2[:, :, -2:]
    coupled_S1 = tf.concat([shared_s2, unshared_s1], -1)
    coupled_S2 = tf.concat([shared_s1, unshared_s2], -1)

    shared_s1 = Flatten()(shared_s1)
    shared_s2 = Flatten()(shared_s2)
    z_shared = tf.concat((tf.expand_dims(shared_s1, axis=1), tf.expand_dims(shared_s2, axis=1)), axis=1)

    deconv_S1 = conv_S1
    deconv_S1 = deconv_S_L2(deconv_S1)
    deconv_S1 = prelu_dec2(deconv_S1)
    deconv_S1 = deconv_S_L3(deconv_S1)

    deconv_S2 = conv_S2
    deconv_S2 = deconv_S_L2(deconv_S2)
    deconv_S2 = prelu_dec2(deconv_S2)
    deconv_S2 = deconv_S_L3(deconv_S2)
    S = tf.concat([tf.expand_dims(deconv_S1, axis=1), tf.expand_dims(deconv_S2, axis=1)], axis=1)

    coupled_S1 = deconv_S_L2(coupled_S1)
    coupled_S1 = prelu_dec2(coupled_S1)
    coupled_S1 = deconv_S_L3(coupled_S1)

    coupled_S2 = deconv_S_L2(coupled_S2)
    coupled_S2 = prelu_dec2(coupled_S2)
    coupled_S2 = deconv_S_L3(coupled_S2)
    coupled_S = tf.concat([tf.expand_dims(coupled_S1, axis=1), tf.expand_dims(coupled_S2, axis=1)], axis=1)

    pae = Model(inputs=input, outputs=S)
    encoder = Model(input, z_shared)
    if enable_summary:
        pae.summary()

    square_diff1 = tf.square(input - coupled_S)
    cp_loss = tf.reduce_mean(square_diff1)

    pae.add_loss(cp_loss)
    pae.add_metric(cp_loss, name='cp_loss', aggregation='mean')

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    pae.compile(optimizer=optimizer)
    return pae, encoder


def prepare_inputs(windows, nr_ae=2, window_size=20):
    new_windows = []
    nr_windows = windows.shape[0]
    for i in range(nr_ae):
        new_windows.append(windows[i * (window_size // 2): nr_windows - (nr_ae - i - 1) * (window_size // 2)])
    return np.transpose(new_windows, (1, 0, 2, 3))


def train_model(windows, enable_summary, window_size, seed, verbose=1, nr_epochs=200, nr_patience=10):
    new_windows = prepare_inputs(windows, nr_ae=2, window_size=window_size)
    pae, encoder = create_parallel_AEs(new_windows, enable_summary, seed)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=nr_patience)

    pae.fit({'data': new_windows},
            epochs=nr_epochs,
            verbose=verbose,
            batch_size=512,
            shuffle=True,
            validation_split=0.0,
            initial_epoch=0,
            callbacks=[callback]
            )

    encoded_windows = encoder.predict(new_windows)
    encoded_windows = np.concatenate((encoded_windows[:, 0, :], encoded_windows[-1:, 1, :]), axis=0)
    return encoded_windows