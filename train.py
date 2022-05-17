import cv2
import glob
import numpy as np
import tensorflow as tf
import os, sys
import tensorflow.keras as keras
import keras.backend as K
from generator import DataGenerator
from LPRnet.LPRnet_separable import LPRnet,CTCLoss,global_context

import wandb
from wandb.keras import WandbCallback

IMAGE_SHAPE = [94,24]
CHARS = "ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789" # exclude I, O
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
DECODE_DICT = {i:char for i, char in enumerate(CHARS)}
NUM_CLASS = len(CHARS)+1

PROJECT_NAME = "LPRnet_keras"
MODEL_PATH = 'trained_models'
TFLITE_PATH = 'tflite_models'

real_images_val = glob.glob('C:\\Users\\carlos\\Desktop\\cs\\ml-sandbox\\ANPR\\LPRnet-keras\\valid\\*\\*.png')
real_images = glob.glob('C:\\Users\\carlos\\Desktop\\cs\\ml-sandbox\\ANPR\\LPRnet-keras\\test\\marty\\*\\*.png')

def main(epochs,MODEL_NAME = "lprnet_straug_twofonts"):
    wandb.init(project=MODEL_NAME, entity="clsandoval")

    if os.path.exists(os.path.join(MODEL_PATH,MODEL_NAME)):
        print("Loading model")
        model = keras.models.load_model(
            os.path.join(MODEL_PATH,MODEL_NAME), custom_objects={"global_context": global_context, "CTCLoss": CTCLoss  }
        )
    else:
        print("Building model from scratch")
        model = LPRnet()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),loss =CTCLoss)
        model.build((1,24,94,3))


    data = []
    labels = []

    for file in real_images:
        label = file.split('\\')[-1].split('_')[0].split('-')[0]
        label = label.replace("O","0")
        image = cv2.imread(file).astype('float32')
        image = cv2.resize(image,(94,24))/256
        data.append(image)
        labels.append([CHARS_DICT[i] for i in label.split('_')[0]])

    training_set = np.array(data,dtype=np.float32)
    training_labels = np.array(labels)
    ragged = tf.ragged.constant(training_labels).to_tensor()
    real_dataset = tf.data.Dataset.from_tensor_slices((training_set,ragged)).batch(64)

    generate = DataGenerator()
    check = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_PATH,MODEL_NAME),
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq=500,
        options=None,
    )
    print("training model for {} epochs".format(epochs))
    model.fit_generator(generator=generate,validation_data=real_dataset,validation_steps=3,epochs=int(epochs),steps_per_epoch=50,callbacks=[WandbCallback(),check])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open("./{}/{}.tflite".format(TFLITE_PATH,MODEL_NAME), 'wb') as f:
      f.write(tflite_model)

if __name__ == "__main__":
    main(sys.argv[1])