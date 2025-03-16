import tensorflow as tf
import numpy as np
import os

# Load the dataset (text corpus)
with open("text_data.txt", "r") as file:
    text = file.read()

# Preprocess the data
chars = sorted(list(set(text)))  # Get all unique characters
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}

# Convert the text into sequences of integers
sequence_length = 100
step = 1
sequences = []
next_chars = []

for i in range(0, len(text) - sequence_length, step):
    sequences.append(text[i:i + sequence_length])
    next_chars.append(text[i + sequence_length])

# Vectorize the sequences and next_chars
X = np.zeros((len(sequences), sequence_length, len(chars)), dtype=np.bool)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(sequence_length, len(chars))),
    tf.keras.layers.Dense(len(chars), activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam")

# Train the model
model.fit(X, y, batch_size=128, epochs=20)

# Function to generate text
def generate_text(model, seed_text, length=100):
    generated = seed_text
    for _ in range(length):
        X_pred = np.zeros((1, sequence_length, len(chars)))
        for t, char in enumerate(seed_text):
            X_pred[0, t, char_to_index[char]] = 1

        prediction = model.predict(X_pred, verbose=0)
        next_char_index = np.argmax(prediction)
        next_char = index_to_char[next_char_index]
        generated += next_char
        seed_text = seed_text[1:] + next_char  # Shift seed text

    return generated

# Generate text using the trained model
seed_text = "The future of artificial intelligence"
generated_text = generate_text(model, seed_text, length=300)
print(generated_text)
