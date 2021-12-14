import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class small_basic_block(keras.layers.Layer):
    def __init__(self,input_shape,out_channels,name=None,**kwargs):
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

class LPRnet(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)


layer = small_basic_block((32,32,3),20)
print(layer.weights)