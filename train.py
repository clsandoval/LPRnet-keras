import cv2
import glob
import numpy as np
import tensorflow as tf
import os, sys
import tensorflow.keras as keras
import keras.backend as K
import argparse
from generator import DataGenerator, RealDataGenerator
from LPRnet.LPRnet_separable import LPRnet,CTCLoss,global_context
from LPRnet.LPRnet_edgeTPU import LPRnet as LPRnet_edgeTPU

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

def main(args):
    MODEL_NAME = args['name']
    epochs = args['epochs']
    wandb.init(
        project="LPRnet-keras",
        entity="clsandoval",
        name=MODEL_NAME,
    )

    if os.path.exists(os.path.join(MODEL_PATH,MODEL_NAME)):
        print("Loading model")
        model = keras.models.load_model(
            os.path.join(MODEL_PATH,MODEL_NAME), custom_objects={"global_context": global_context, "CTCLoss": CTCLoss  }
        )
    else:
        print("Building model from scratch")
        if args['arch'] != "edgetpu":
            model = LPRnet()
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),loss =CTCLoss)
            model.build((1,24,94,3))
        else:
            model = LPRnet_edgeTPU()
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),loss =CTCLoss)

    data = []
    labels = []

    for file in real_images_val:
        label = file.split('\\')[-1].split('_')[0].split('-')[0]
        label = label.replace("O","0")
        image = cv2.imread(file).astype('float32')
        image = cv2.resize(image,(94,24))/256
        data.append(image)
        labels.append([CHARS_DICT[i] for i in label.split('_')[0]])

    training_set = np.array(data,dtype=np.float32)
    training_labels = np.array(labels)
    ragged = tf.ragged.constant(training_labels).to_tensor()
    val_dataset = tf.data.Dataset.from_tensor_slices((training_set,ragged)).batch(64).repeat()

    generate = DataGenerator()
    if args['gen'] == "real":
        print("Real images for training")
        generate = RealDataGenerator()
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
    model.fit_generator(generator=generate,validation_data=val_dataset,validation_steps=50,epochs=int(epochs),steps_per_epoch=50,callbacks=[WandbCallback(),check])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open("./{}/{}.tflite".format(TFLITE_PATH,MODEL_NAME), 'wb') as f:
      f.write(tflite_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-e','--epochs', help='Number of epochs', required=True)
    parser.add_argument('-a','--arch', help='Architecture to use', required=True)
    parser.add_argument('-n','--name', help='Model name', required=True)
    parser.add_argument('-g', '--gen', help='Dataset geeneration type', required=True)
    args = vars(parser.parse_args())
    main(args)