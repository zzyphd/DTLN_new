# importing libs
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.models import load_model
import librosa

data = np.load("data.npy")

binData = float(data).to_bytes(1, 'big')
print(binData)