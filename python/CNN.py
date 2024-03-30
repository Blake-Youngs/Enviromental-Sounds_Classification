import pandas as pd
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM, Dropout, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from scipy.io import wavfile



def CNN():

    model = Sequential()  # Using the sequential model that is modular, you can keep adding layers
    # https://keras.io/api/layers/convolution_layers/convolution2d/
    # model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape= input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.5))
    model.add(Flatten)
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["acc"])

    return model

# Getting the data from a csv file

df = pd.read_csv('Audio_Files/samples_data.csv')

# lets just plot the data

df.set_index('filename', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('audio/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.category))
class_dist = df.groupby(['category'])['length'].mean()

n_samples =




fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()
