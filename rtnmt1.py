from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow_addons import metrics as mt
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


def make_predict(img_pth):
    F1Score = mt.F1Score(num_classes=2, threshold=.5)

    model = load_model('/home/tim/PycharmProjects/DataScience/computer_vision/RTMSD/model.h5',
     compile=False, custom_objects={'F1Score':F1Score})
    """
    # img = load_img('/home/tim/Downloads/pexels-mathias-pr-reding-4646843.jpg', target_size=(224, 224))
    img = load_img(img_pth, target_size=(224, 224))
    img = img_to_array(img)
    img = tf.expand_dims(img, 0)
    print(img.shape)
    img = img/255.0

    pred = model.predict(img)
    """
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img_pth
    # image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    # pred = np.round(tf.sigmoid(pred))
    # 'mask' if np.argmax(pred) == 0 else 'no mask'
    return np.argmax(prediction)
