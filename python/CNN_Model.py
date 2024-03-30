import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import split_csv_file as scf

# Split data into training and testing sets
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
X_train, X_test, y_train_encoded, y_test_encoded = scf.split_csv_file('../data/sounds/ESC-50-master/meta/esc50.csv')

# Define CNN model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=(128, 431, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(50, activation='softmax'))  # Adjusted output units to match 50 possible outputs

model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_encoded, epochs=20, batch_size=32, validation_data=(X_test, y_test_encoded))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print('Test accuracy:', test_acc)
