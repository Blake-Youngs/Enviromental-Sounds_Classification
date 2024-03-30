from sklearn.model_selection import train_test_split
import generate_mel_spectrogram as gms
import numpy as np
import pandas as pd
from keras.utils import to_categorical


def split_csv_file(csv_file_path, test_size=0.2):
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Initialize lists to hold spectrogram images and labels
    X = []
    y = []

    # Load audio files and generate spectrograms
    for index, row in df.iterrows():
        file_name = row['filename']
        label = row['target']
        spectrogram = gms.generate_mel_spectrogram('../data/sounds/ESC-50-master/audio/' + file_name)
        X.append(spectrogram)
        y.append(label)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # One-hot encode the labels
    y_encoded = to_categorical(y, num_classes=50)

    # Split data into training and testing sets
    return train_test_split(X, y_encoded, test_size=test_size, random_state=42)