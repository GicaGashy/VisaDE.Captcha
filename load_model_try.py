import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from keras import layers


data_dir = Path('all')
max_length = 6


images = sorted(list(map(str, list(data_dir.glob("*.png")))))

labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
characters = set(char for label in labels for char in label)
model = keras.models.load_model('model/')

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for x in images:
        pred = model.predict(x)
        pred_txt = decode_batch_predictions(pred)

        print(x)
    print(model)


