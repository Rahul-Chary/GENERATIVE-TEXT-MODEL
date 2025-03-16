# GENERATIVE-TEXT-MODEL

COMPANY: CODTECH IT SOLUTIONS

NAME: KONDAMADUGU RAHUL CHARY

INTERN ID: CT12ONT

DOMAIN: ARTIFICIAL INTELLIGENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

Description:-

To create a text generation model using GPT (like GPT-4) or LSTM (Long Short-Term Memory), we can break the task into the following steps. Below is an outline for both models:

1. Setting Up the Environment
You'll need an environment with Python, necessary libraries, and computational resources. For large models like GPT, you may need access to a cloud service (e.g., AWS, GCP, or Azure) with GPUs.

Install the necessary libraries:

bash
Copy
pip install tensorflow torch transformers numpy pandas
2. Using GPT for Text Generation
Using pre-trained GPT models (like GPT-2 or GPT-3) is straightforward with libraries like transformers by Hugging Face. Here's how to implement it.

GPT-2 Model Example:
python
Copy
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium", "gpt2-large", or "gpt2-xl" for larger models
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Encode input prompt
input_text = "The future of artificial intelligence"
inputs = tokenizer.encode(input_text, return_tensors="pt")

# Generate text
output = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7, top_k=50, top_p=0.9)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
Parameters:
max_length: Controls the length of the generated text.
num_return_sequences: Number of different completions to generate.
temperature: Controls the randomness of predictions (higher = more random).
top_k and top_p: Control how many potential words to consider at each step.
3. Using LSTM for Text Generation
For LSTM-based generation, you’ll need to prepare a dataset, preprocess it, and train an LSTM model.

Example Code for LSTM Text Generation (using TensorFlow/Keras):
python
Copy
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
Steps:
Prepare the Data: Preprocess the input text into sequences of characters (e.g., 100 characters per sequence).
Build the LSTM Model: Use an LSTM layer with a dense softmax layer at the end to predict the next character in the sequence.
Train the Model: Use the sequences to train the model with categorical cross-entropy loss.
Generate Text: After training, input a seed text, and the model will predict the next character, generating coherent text.
4. Topic-specific Generation
For both GPT and LSTM models, generating coherent paragraphs on specific topics can be achieved by:

For GPT: Provide a prompt that is related to the specific topic you want to generate text about. For example:

python
Copy
input_text = "Artificial intelligence in healthcare is transforming the medical industry. Some major innovations include"
For LSTM: The model needs to be trained on a corpus that’s specifically relevant to the desired topic. For instance, if you want to generate text about AI in healthcare, you would use a text corpus on healthcare AI innovations.

5. Considerations
Data Quality: The quality of the text generated depends heavily on the quality and relevance of the dataset. If using GPT, the model has been pre-trained on diverse data, but fine-tuning with your specific topic dataset can enhance results.
Computational Resources: GPT-based models (especially larger ones like GPT-3 or GPT-4) require significant computational power. LSTM models are less resource-intensive but may not generate as coherent text as GPT models.



