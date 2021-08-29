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


data = np.load('data.npy')

# in_data,fs = librosa.core.load("data/total.wav", sr=16000, mono=True)
# print(in_data.shape)

M = 512
k = np.log2(M)
k = int(k)
print ('M:',M,'k:',k)

train_Data = data[:1000000, :]
val_Data = data[1000000:, :]


R = 4/7
n_channel = 512
print (int(k/R))
input_signal = Input(shape=(M,))
encoded = Dense(M, activation='relu')(input_signal)
encoded1 = Dense(n_channel, activation='linear')(encoded)
encoded2 = BatchNormalization()(encoded1)

EbNo_train = 5.01187 #  coverted 7 db of EbNo
encoded3 = GaussianNoise(np.sqrt(1/(2*R*EbNo_train)))(encoded2)

decoded = Dense(M, activation='relu')(encoded3)
decoded1 = Dense(M, activation='softmax')(decoded)

autoencoder = Model(input_signal, decoded1)
#sgd = SGD(lr=0.001)
autoencoder.compile(optimizer='adam', loss="mse")

autoencoder.fit(data, data,
                epochs=10,
                batch_size=300,
                validation_data=(val_Data, val_Data))

autoencoder.save('autoencoder.model')

# encoder = Model(autoencoder.input, autoencoder.get_layer().output)

encoded_input = Input(shape=(n_channel,))
deco = autoencoder.layers[-2](encoded_input)
deco = autoencoder.layers[-1](deco)
# create the decoder model
decoder = Model(encoded_input, deco)
