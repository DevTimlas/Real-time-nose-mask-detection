import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAvgPool2D, Flatten, Dense, Dropout
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import *


IMG_SHAPE = (224, 224, 3)
train_path = '/home/tim/Datasets/face-mask-dataset/train/'
valid_path = '/home/tim/Datasets/face-mask-dataset/test/'

train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_data = train_datagen.flow_from_directory(train_path,
                                               batch_size=15,
                                               target_size=(224, 224))

val_datagen = ImageDataGenerator(rescale=1.0/255, )

valid_data = val_datagen.flow_from_directory(valid_path,
                                             batch_size=15,
                                             target_size=(224, 224))

class_names = list(train_data.class_indices)
print('class_names', class_names)

base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3))

image_batch, label_batch = next(iter(train_data))
feature_batch = base_model(image_batch)

gal = tf.keras.layers.GlobalAvgPool2D()
feature_batch_average = gal(feature_batch)

pl = tf.keras.layers.Dense(len(class_names))
pb = pl(feature_batch_average)

inputs = tf.keras.Input(IMG_SHAPE)
x = base_model(inputs, training=False)
x = gal(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = pl(x)

model = tf.keras.Model(inputs, outputs)
print(model.summary())

blr = 1e-4
f1_score = F1Score(num_classes=len(class_names), threshold=0.5)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=blr),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', f1_score])

epochs = 5

tb = TensorBoard(log_dir='/home/tim/trained/mobile_netV2/RTFMD/logs/',)
mc = ModelCheckpoint(f'/home/tim/trained/mobile_netV2/RTFMD/model/model_{epochs}',
                     monitor='accuracy', save_best_only=True)
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
logger = CSVLogger('/home/tim/trained/mobile_netV2/RTFMD/csv_logger.csv',
                   append=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=.2, min_lr=1e-2)

model.fit(train_data, validation_data=valid_data, epochs=epochs,
          callbacks=[tb, mc, es, logger, reduce_lr])
