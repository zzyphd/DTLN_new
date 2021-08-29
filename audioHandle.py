
import scipy.io.wavfile as wav
import os
import glob
import numpy as np

def merge_files(path_read_folder, path_write_wav_file):
    #
    files = os.listdir(path_read_folder)
    merged_signal = []
    for filename in glob.glob(os.path.join(path_read_folder, '*.wav')):
        # print(filename)
        sr, signal = wav.read(filename)
        merged_signal.append(signal)
    merged_signal = np.hstack(merged_signal)
    merged_signal = np.asarray(merged_signal, dtype=np.int16)
    wav.write(path_write_wav_file, sr, merged_signal)


# noisy train total
path_read_folder = "data1"
path_write_wav_file = "data/total.wav"
merge_files(path_read_folder, path_write_wav_file)
