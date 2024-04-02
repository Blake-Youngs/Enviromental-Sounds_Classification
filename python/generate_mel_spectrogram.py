import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


def visualize_spectrogram(interpolated_log_mel_spectrogram, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(interpolated_log_mel_spectrogram, sr=sr, hop_length=512, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Interpolated Log Mel Spectrogram')
    # plt.title('Log Mel Spectrogram')
    plt.tight_layout()
    plt.savefig('output_file')
    plt.show()


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
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512, n_mels=60)

    # Convert to log scale
    log_mel_spectrogram = librosa.power_to_db(spectrogram)

    return log_mel_spectrogram


def interpolated_log_mel_spectrogram(wave_file_path, rate=None, visual_frame_rate=1, audio_frame_rate=5):
    # Load audio file
    audio, sr = librosa.load(wave_file_path, sr=rate)  # sr=None to preserve the native sampling rate

    # Compute Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512, n_mels=60)

    # Convert to log scale
    log_mel_spectrogram = librosa.power_to_db(spectrogram)

    # Compute derivative of log mel spectrogram
    delta_log_mel_spectrogram = librosa.feature.delta(log_mel_spectrogram)

    # Interpolate both channels
    visual_frames = log_mel_spectrogram.shape[1]
    audio_frames = len(audio) // (sr // audio_frame_rate)
    interpolated_log_mel_spectrogram = np.zeros((log_mel_spectrogram.shape[0], visual_frames))
    interpolated_delta_log_mel_spectrogram = np.zeros((delta_log_mel_spectrogram.shape[0], visual_frames))

    for i in range(visual_frames):
        audio_index = min(int(i * (audio_frames / visual_frames)), audio_frames - 1)
        interpolated_log_mel_spectrogram[:, i] = log_mel_spectrogram[:, audio_index]
        interpolated_delta_log_mel_spectrogram[:, i] = delta_log_mel_spectrogram[:, audio_index]
    return interpolated_log_mel_spectrogram


# visualize_spectrogram(interpolated_log_mel_spectrogram('../data/sounds/ESC-50-master/audio/2-70280-A-18.wav'), 44100)
