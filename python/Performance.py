import generate_mel_spectrogram as gms
import tensorflow as tf
import pandas as pd
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('C:\\Users\\Herib\\Downloads\\Enviromental-Sounds_Classification-main\\Enviromental-Sounds_Classification-main\\python\\files\\10Label_737.h5')
# Load the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\Herib\\Downloads\\Enviromental-Sounds_Classification-main\\Enviromental-Sounds_Classification-main\\python\\files\\esc50-10.csv')

# Create a dictionary mapping target labels to categories
label_to_category = dict(zip(df['target'], df['category']))

# Define your labels
labels = ['helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw']

# Need to get the wav file data
log_mel_spec = gms.log_mel_spectrogram('C:\\Users\\Herib\\Downloads\\Enviromental-Sounds_Classification-main\\Enviromental-Sounds_Classification-main\\python\\test audio\\Siren.wav', 8000)

# Reshape the input to match the model's input shape
log_mel_spec = log_mel_spec.reshape(1, log_mel_spec.shape[0], log_mel_spec.shape[1], 1)

# Resize or pad the spectrogram to match the desired shape (60, 61)
log_mel_spec_resized = np.resize(log_mel_spec, (1, 60, 61, 1))

# Make predictions using the loaded model
predictions = model.predict(log_mel_spec_resized)

# Get the predicted label index
predicted_label_index = tf.argmax(predictions[0]).numpy()

# Get the predicted label and category
predicted_label = labels[predicted_label_index]
predicted_category = label_to_category.get(predicted_label)

print("Predicted label index:", predicted_label_index)
print("Label to Category mapping:")
for label, category in label_to_category.items():
    print(f"Label: {label}, Category: {category}")

