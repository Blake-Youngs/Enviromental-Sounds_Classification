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
        spectrogram = gms.log_mel_spectrogram('../data/sounds/ESC-50-master/audio/' + file_name, rate=None)
        X.append(spectrogram)
        y.append(label)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # One-hot encode the labels for 20 classes
    num_classes = 20
    y_train_encoded = to_categorical(y_train, num_classes=num_classes)
    y_test_encoded = to_categorical(y_test, num_classes=num_classes)

    return X_train, X_test, y_train_encoded, y_test_encoded
