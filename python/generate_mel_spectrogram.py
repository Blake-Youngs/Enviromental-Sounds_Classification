import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

from glob import glob

all_files = glob('../data/sounds/ESC-50-master/audio/*')


def generate_mel_spectrogram(wave_file_path, rate=None):
    # Load audio file
    audio, sr = librosa.load(wave_file_path, sr=rate)  # sr=None to preserve the native sampling rate

    # Compute spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

    # Convert to decibels (log scale)]
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    return spectrogram_db
