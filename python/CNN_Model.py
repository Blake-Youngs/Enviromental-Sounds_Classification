import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import split_csv_file as scf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force TensorFlow to use CPU only

# Split data into training and testing sets
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
X_train, X_test, y_train, y_test = scf.split_csv_file('../data/sounds/ESC-50-master/meta/esc50-50.csv')

# Convert lists to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Get the shape of the training data
num_samples, sequence_length, width = X_train.shape
print(sequence_length, width)

# One-hot encode the labels
num_classes = 50  # Update to the correct number of classes
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes)

# Define CNN model architecture
model = Sequential()

# Layer 1
model.add(Conv2D(24, (5, 5), activation='relu', input_shape=(sequence_length, width, 1)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

# Layer 2
model.add(Conv2D(36, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 3
model.add(Conv2D(48, (3, 3), activation='relu'))

# Layer 4
model.add(Flatten())
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.5))

# Layer 5
model.add(Dense(10, activation='softmax'))

model.summary()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_encoded, epochs=200, validation_data=(X_test, y_test_encoded))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print('Test accuracy:', test_acc)
