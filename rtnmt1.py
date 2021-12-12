from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow_addons import metrics as mt
import tensorflow as tf
import numpy as np

F1Score = mt.F1Score(num_classes=2, threshold=.5)

model = load_model('/home/tim/PycharmProjects/DataScience/computer_vision/RTMSD/model.h5',
 compile=False, custom_objects={'F1Score':F1Score})
img = load_img('/home/tim/Downloads/pexels-pixabay-220453_without_mask.jpg', target_size=(224, 224))
img  = img_to_array(img)
img = tf.expand_dims(img, 0)
print(img.shape)
img = img/255.0

pred = model.predict(img)

# pred = np.round(tf.sigmoid(pred))
print('mask' if np.argmax(pred)==0 else 'no mask')
