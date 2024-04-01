import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape
import split_csv_file as scf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force TensorFlow to use CPU only

# Assuming x_train shape is (640, 128, 431)
input_shape = (128, 431, 1)  # Adjusted input shape

# Split data into training and testing sets
X_train, X_test, y_train_encoded, y_test_encoded = scf.split_csv_file('../data/sounds/ESC-50-master/meta/esc50.csv')

# Define CNN model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(20, activation='softmax'))  # Adjusted output units to match 20 possible outputs

model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Reshape X_train and X_test to match the expected input shape
X_train = X_train.reshape(-1, 128, 431, 1)
X_test = X_test.reshape(-1, 128, 431, 1)

# Train the model
history = model.fit(X_train, y_train_encoded, epochs=20, batch_size=32, validation_data=(X_test, y_test_encoded))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print('Test accuracy:', test_acc)
