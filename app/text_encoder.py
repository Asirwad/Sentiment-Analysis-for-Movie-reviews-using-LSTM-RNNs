from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing import sequence


# Get the word index from the IMDb dataset
word_index = imdb.get_word_index()
MAX_LEN = 250


def encode_text(text):
    # Convert the text to a list of lowercase words and remove punctuation
    # It will tokenize the text into individual words
    tokens = text_to_word_sequence(text)

    # initialize an empty list to store the encoded representation of the text
    encoded_text = []

    # Encode each word in the text using the word index, or use 0 if the word is not found in the word index
    for word in tokens:
        # If the word exists in the word index, get its corresponding integer value from the word index
        # If the word is not in the word index, use 0 as a placeholder for unknown words
        word_index_value = word_index.get(word, 0)
        encoded_text.append(word_index_value)

    # Pad the encoded text to ensure all sequences have the same length
    # This is necessary to create a consistent input shape for the neural network
    # The 'MAX_LEN' variable determines the maximum length of the sequences
    padded_encoded_text = sequence.pad_sequences([encoded_text], MAX_LEN)[0]
    return padded_encoded_text
