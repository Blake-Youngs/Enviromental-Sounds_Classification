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
        spectrogram = gms.test_spectrogram('../data/sounds/ESC-50-master/audio/' + file_name, rate=7000)
        X.append(spectrogram)
        y.append(label)

    # Split data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test
