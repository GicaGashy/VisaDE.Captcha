import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras_preprocessing import image

characters = ['a', 'b', 'c', 'd']

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [50, 300])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}

data_dir = Path('all')
max_length = 6
#
#
# images = sorted(list(map(str, list(data_dir.glob("*.png")))))
#
# labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
# characters = set(char for label in labels for char in label)

#
#

#
# def decode_batch_predictions(pred):
#     input_len = np.ones(pred.shape[0]) * pred.shape[1]
#     # Use greedy search. For complex tasks, you can use beam search
#     results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
#         :, :max_length
#     ]
#     # Iterate over the results and get back the text
#     output_text = []
#     for res in results:
#         res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
#         output_text.append(res)
#     return output_text

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#
#     print(model)

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

# test_image = image.load_img("all/2a5ym7.png", color_mode="grayscale", target_size=(300, 50))
test_image = image.load_img("all/2a5ym7.png", color_mode="grayscale", )
print("Just loading---start")
print(test_image)
print("Just loading---end")
print("-------------")
print("img_to_array---start")
test_image = image.img_to_array(test_image)

print(test_image)
print("img_to_array---end")
print("---------------")
print("expand_dims --- start")
test_image = np.array([test_image])
test_image = np.transpose(test_image)
print(test_image)
print("expand_dims --- end")


model = keras.models.load_model('model/')
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
# model = model.fit(test_image, "asdfff")
result = prediction_model.predict(test_image)
pred_text = decode_batch_predictions(result)
print(pred_text)



