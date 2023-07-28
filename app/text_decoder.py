from tensorflow.keras.datasets import imdb

# Get the word index from the IMDb dataset
word_index = imdb.get_word_index()
MAX_LEN = 250

# reverse word index to map integer values back to their respective words
reversed_word_index = {value: key for (key, value) in word_index.items()}


def decode_integers(integers):
    # Define the PAD value used for padding sequences (0 in this case)
    pad = 0
    decoded_text = ""

    # Iterate through each integer in the 'integers' list
    for num in integers:
        # Skip the PAD value since it was used for padding and does not represent any word
        if num != pad:
            decoded_word = reversed_word_index[num]
            # Append the decoded word to the 'text' string with a space separator
            decoded_text += decoded_word + ' '

    return decoded_text.strip()