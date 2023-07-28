import numpy as np
from tensorflow.keras.models import load_model

from app.text_encoder import encode_text
from app.text_decoder import decode_integers


def predict(input_text):
    model = load_model('models/movie_review_rnn.h5')
    encoded_text = encode_text(text=input_text)

    # reshape the encoded text to the model compactable form
    reshaped_encoded_text = np.zeros((1, 250))
    reshaped_encoded_text[0] = encoded_text

    result = model.predict(reshaped_encoded_text)
    result = result[0][0]
    print("Satisfaction : ", result*100, "%")
    if result <= 0.25:
        print("Negative review")
    else:
        print("positive review")


text = input("Enter something :")
predict(input_text=text)


