#%%
import numpy as np, tensorflow as tf, tensorflow.keras as keras, tensorflow.keras.layers as layers


"""
LocNet implemented for LPRnet as described in https://arxiv.org/abs/1506.02025

To-Do
    - Localization Network
    - Grid Generator
    - Sampler

"""

#
def build_localization_layers(input_layer):
    x = layers.Conv2D(8,(7,7), padding='same') (input_layer)
    x = layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(10,(5,5), padding='same')(x)
    localization = layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same')(x)
    return localization


def build_regressor_layers(input_layer):
    x = layers.Dense(32, activation='relu')(input_layer)
    x1 = layers.Dense(6, weights= [np.zeros(shape=(32,6)),np.array([1,0,0,0,1,0])])(x)
    return x1

input_layer = layers.Input(shape=(24,94,3))
localization = build_localization_layers(input_layer)
reshape_layer = layers.Reshape((-1,6*24*10))(localization)
regressor = build_regressor_layers(reshape_layer)
theta = layers.Reshape((-1,2,3))(regressor)
model = keras.Model(inputs=input_layer, outputs=theta)
model.summary()
#%%
