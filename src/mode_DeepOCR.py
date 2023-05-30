import h5py

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, LSTM, \
    BatchNormalization, Activation, Flatten, Embedding, ReLU,GlobalAveragePooling1D, Input, concatenate
from tensorflow.keras import backend as K
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

def mode_DeepOCR():
    def DW_block(X, kernels, size):
        out = tf.keras.layers.SeparableConv1D(kernels, size,depth_multiplier=1, padding='same',data_format="channels_last")(
            X)
        #out = tf.keras.layers.Conv1D(kernels, size, padding='same')(X)
        out = tf.keras.layers.ReLU()(out)
        #add
        #out = Dropout(rate=0.5)(out)
        X_c = tf.keras.layers.Conv1D(kernels, 1, padding='same')(X)
        #out = tf.keras.layers.Conv1D(300, 1, padding='same')(out)
        out = tf.keras.layers.add([X_c, out])
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.MaxPool1D(pool_size=3, strides=3)(out)
        out = Dropout(rate=0.5)(out)
        return out

    inputs = tf.keras.layers.Input([1000, 4])
    x = tf.keras.layers.Conv1D(filters=300, kernel_size=19, padding="same")(inputs)
    x = DW_block(x, kernels=200, size=11)
    x = DW_block(x, kernels=100, size=9)
    x_output = tf.keras.layers.GlobalAveragePooling1D()(x)
    #add
    x_output =  tf.keras.layers.Dense(300, activation='relu')(x_output)
    x_output = Dropout(rate=0.6)(x_output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x_output)
    model = tf.keras.Model(inputs=inputs, outputs=output, name="DeepOCR")
    return model