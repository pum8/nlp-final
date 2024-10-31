import pandas as pd

# Load the dataset to examine its structure and understand its contents
file_path = '/mnt/data/AQ_Platinum_all.csv'
data = pd.read_csv(file_path)

# Display the first few rows and summary information to understand the data structure
data.head(), data.info()

from sklearn.model_selection import train_test_split
import numpy as np

# Data preprocessing
# Convert timestamp to datetime for easier handling and extract relevant temporal features

data['timestamp'] = pd.to_datetime(data['_last_judgment_at'], errors='coerce')
data['timestamp'] = data['timestamp'].fillna(method='ffill')  # Fill missing timestamps forward as an assumption

# Extract verb and context features for modeling
data['text_data'] = data['before'].fillna('') + ' ' + data['verb'].fillna('') + ' ' + data['after'].fillna('')
text_data = data['text_data']
timestamps = data['timestamp']

# Encode labels as temporal sequences
# For simplicity, assume 'verb' is the event label, we can assign an index to each event

# Filter out necessary columns and drop rows with missing data in critical columns
data_filtered = data[['timestamp', 'text_data']].dropna()

# Split the data into train and test sets, maintaining temporal order by timestamp
data_filtered = data_filtered.sort_values(by='timestamp')
train_data, test_data = train_test_split(data_filtered, test_size=0.2, shuffle=False)

# Display sample of processed data
train_data.head()

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['text_data'])
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_data['text_data'])
test_sequences = tokenizer.texts_to_sequences(test_data['text_data'])

# Pad sequences for uniform input size
max_length = max(len(seq) for seq in train_sequences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Define LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()
