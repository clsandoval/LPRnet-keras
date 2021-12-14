import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

IMAGE_SHAPE = [94,24]
CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789" # exclude I, O
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
DECODE_DICT = {i:char for i, char in enumerate(CHARS)}
NUM_CLASS = len(CHARS)+1

class small_basic_block(keras.layers.Layer):

    def __init__(self,out_channels,name=None,**kwargs):
        super().__init__(**kwargs)
        out_div4=int(out_channels/4)
        self.main_layers = [
            keras.layers.Conv2D(out_div4,(1,1),padding='same',activation='relu'),
            keras.layers.Conv2D(out_div4,(1,3),padding='same',activation='relu'),
            keras.layers.Conv2D(out_div4,(3,1),padding='same',activation='relu'),
            keras.layers.Conv2D(out_div4,(1,1),padding='same',activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ]
    
    def call(self,input):
        x = input
        for layer in self.main_layers:
            x = layer(x)
        return x

class LPRnet(keras.Model):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.cnn_layers= [
            keras.layers.Conv2D(64,(3,3),padding='same',name='main_conv1'),
            #keras.layers.BatchNormalization(),
            #keras.layers.ReLU(),
            #keras.layers.MaxPool2D(pool_size=(3,3),strides=1),
            #small_basic_block(128),
            #keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,1)),
            #small_basic_block(256),
            #small_basic_block(256),
            #keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,1)),
            #keras.layers.Dropout(0.5),
            #keras.layers.Conv2D(256,(1,4),strides=1,padding='same',name='main_conv2'),
            #keras.layers.Dropout(0.5),
            #keras.layers.Conv2D(NUM_CLASS,(13,1),padding='same',name='main_conv3'),  
            #keras.layers.BatchNormalization(),
            #keras.layers.ReLU(),
        ]

    def call(self,inputs,training=False):
        x = inputs
        for layer in self.cnn_layers:
            x = layer(x)
        return x
