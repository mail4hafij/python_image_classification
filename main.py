import tensorflow as tf
import numpy as np


filename = 'images/cat.jpg'
img = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
img_to_array = tf.keras.preprocessing.image.img_to_array(img)
# need fourth dimension for the algorithm to predict
final_image = np.expand_dims(img_to_array, axis=0)
final_image = tf.keras.applications.mobilenet.preprocess_input(final_image)

# Loading the deep learning model
# deep_learning_alg = tf.keras.applications.mobilenet.MobileNet()
deep_learning_alg = tf.keras.applications.mobilenet_v2.MobileNetV2()
predictions = deep_learning_alg.predict(final_image)

# decoding the predictions
for decoded_prediction in tf.keras.applications.imagenet_utils.decode_predictions(predictions):
    for name, desc, score in decoded_prediction:
        print('- {} ({:.2f}%)'.format(desc, 100 * score))


