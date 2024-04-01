import librosa
import numpy as np
import librosa.display


def mel_spectrogram(wave_file_path, rate=None):
    # Load audio file
    audio, sr = librosa.load(wave_file_path, sr=rate)  # sr=None to preserve the native sampling rate

    # Compute spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

    return spectrogram


def log_mel_spectrogram(wave_file_path, rate=None):
    # Load audio file
    audio, sr = librosa.load(wave_file_path, sr=rate)  # sr=None to preserve the native sampling rate

    # Compute Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

    # Convert to log scale
    log_mel_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    return log_mel_spectrogram
