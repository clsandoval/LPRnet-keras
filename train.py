import cv2
import glob
import numpy as np
import tensorflow as tf
import os
import tensorflow.keras as keras
import keras.backend as K
from generator import DataGenerator
from model_depthwise import LPRnet,CTCLoss,global_context

import wandb
from wandb.keras import WandbCallback

IMAGE_SHAPE = [94,24]
CHARS = "ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789" # exclude I, O
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
DECODE_DICT = {i:char for i, char in enumerate(CHARS)}
NUM_CLASS = len(CHARS)+1

real_images_val = glob.glob('C:\\Users\\carlos\\Desktop\\cs\\ml-sandbox\\ANPR\\LPRnet-keras\\valid\\*\\*.png')
real_images = glob.glob('C:\\Users\\carlos\\Desktop\\cs\\ml-sandbox\\ANPR\\LPRnet-keras\\test\\marty\\*\\*.png')

def main(MODEL_NAME = "depthwise_model_rabdomchars_perspective"):
    wandb.init(project=MODEL_NAME, entity="clsandoval")
    wandb.config = {
    "learning_rate": 0.001,
    "epochs": 400,
    "batch_size": 64
    }

    if os.path.exists(MODEL_NAME):
        model = keras.models.load_model(
            MODEL_NAME, custom_objects={"global_context": global_context, "CTCLoss": CTCLoss  }
        )
    else:
        model = LPRnet()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),loss =CTCLoss)
        model.build((1,24,94,3))


    data = []
    labels = []

    for file in real_images:
        label = file.split('\\')[-1].split('_')[0].split('-')[0]
        label = label.replace("O","0")
        image = cv2.imread(file,cv2.IMREAD_COLOR)
        image = cv2.resize(image,(94,24))/256
        data.append(image)
        labels.append([CHARS_DICT[i] for i in label.split('_')[0]])

    training_set = np.array(data,dtype=np.float32)
    training_labels = np.array(labels)
    ragged = tf.ragged.constant(training_labels).to_tensor()
    real_dataset = tf.data.Dataset.from_tensor_slices((training_set,ragged)).batch(64)

    generate = DataGenerator()
    check = tf.keras.callbacks.ModelCheckpoint(
        './{}'.format(MODEL_NAME),
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq=500,
        options=None,
    )
    model.fit_generator(generator=generate,validation_data=real_dataset,validation_steps=5,epochs=10000,steps_per_epoch=50,callbacks=[WandbCallback(),check])