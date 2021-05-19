import tensorflow as tf
import numpy as np


filename = 'images/cat.jpg'
# using tensorflow
img = tf.keras.preprocessing.image.load_img(filename, target_size = (224, 224))

# Loading the deep learning model
# deep_learning_alg = tf.keras.applications.mobilenet.MobileNet()
deep_learning_alg = tf.keras.applications.mobilenet_v2.MobileNetV2()
# adding 4th dimension
final_image = np.expand_dims(img, axis = 0)
final_image = tf.keras.applications.mobilenet.preprocess_input(final_image)
predictions = deep_learning_alg.predict(final_image)
# decoding the predictions
results = tf.keras.applications.imagenet_utils.decode_predictions(predictions)
print(results)