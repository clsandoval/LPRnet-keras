import numpy as np
from gen_plates_keras import *
import tensorflow as tf  

gen = ImageGenerator()
IMAGE_SHAPE = [94,24]
CHARS = "ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789" # exclude I, O
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
DECODE_DICT = {i:char for i, char in enumerate(CHARS)}
NUM_CLASS = len(CHARS)+1

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self):
        pass

    def __len__(self):
        return 50

    def __getitem__(self,index):
        data, labels = gen.generate_images(64)
        gen_labels = []
        for label in labels:
            gen_labels.append([CHARS_DICT[i] for i in label.split('_')[0]])
        pics =np.array(data)
        labels = np.array(labels)
        training_set = np.array(pics,dtype=np.float32)
        training_labels = np.array(gen_labels)
        ragged = tf.ragged.constant(training_labels).to_tensor()
        return training_set,ragged