from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras import Sequential


def get_model(vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=32),
        LSTM(units=32),
        Dense(units=1, activation='sigmoid')
    ])
    return model