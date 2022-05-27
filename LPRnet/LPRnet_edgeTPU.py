#%%
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers
import keras.backend as K
#from generator import DataGenerator

IMAGE_SHAPE = [94,24]
CHARS = "ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789" # exclude I, O
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
DECODE_DICT = {i:char for i, char in enumerate(CHARS)}
NUM_CLASS = len(CHARS)+1


def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    
    return loss

def smallblock(out_channels,inputs):
    out_div4=int(out_channels/4)
    main_layers = [
        keras.layers.Conv2D(out_div4,(1,1),padding='same',kernel_initializer=keras.initializers.glorot_uniform(),bias_initializer=keras.initializers.constant()),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(out_div4,(3,1),padding='same',kernel_initializer=keras.initializers.glorot_uniform(),bias_initializer=keras.initializers.constant()),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(out_div4,(1,3),padding='same',kernel_initializer=keras.initializers.glorot_uniform(),bias_initializer=keras.initializers.constant()),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(out_channels,(1,1),padding='same',kernel_initializer=keras.initializers.glorot_uniform(),bias_initializer=keras.initializers.constant()),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
    ]  
    x = inputs
    for layer in main_layers:
        x = layer(x)
    return x


def LPRnet():
    #main network layers
    input_layer = tf.keras.Input(shape=(24, 94, 3))
    x_1 = keras.layers.Conv2D(64,kernel_size = (3,3),strides=1,padding='same',name='main_conv1')(input_layer)
    x = keras.layers.BatchNormalization(name='BN1')(x_1)
    x = keras.layers.ReLU(name='RELU1')(x)
    x = keras.layers.MaxPool2D(pool_size=(3,3),strides=(1,1),name='maxpool2d_1',padding='same')(x)
    x_2 = smallblock(128,x)
    x = keras.layers.MaxPool2D(pool_size=(3,3),strides=(1,2),name='maxpool2d_2',padding='same')(x_2)
    x_3 = smallblock(256,x)
    x_4 = smallblock(256,x_3)
    x = keras.layers.MaxPool2D(pool_size=(3,3),strides=(1,2),name='maxpool2d_3',padding='same')(x_4)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(256,(4,1),strides=1,padding='same',name='main_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(NUM_CLASS,(1,13),padding='same',name='main_conv3')(x)
    x = keras.layers.BatchNormalization()(x)
    x_5 = keras.layers.ReLU()(x)
   
    #global context layers with early fusion
    #avg pool -> x/mean(x^2) -> concatenate
    avg_pool_1 = keras.layers.AveragePooling2D(pool_size=(1,4),strides=(1,4),padding='same')(x_1)
    sq_1 = keras.layers.Multiply()([avg_pool_1,avg_pool_1])
    sqm_1 = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x))(sq_1)
    gc_1 = keras.layers.Lambda(lambda x: x[0]/x[1])([avg_pool_1 ,sqm_1])

    avg_pool_2 = keras.layers.AveragePooling2D(pool_size=(1,4),strides=(1,4),padding='same')(x_2)
    sq_2 = keras.layers.Multiply()([avg_pool_2,avg_pool_2])
    sqm_2 = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x))(sq_2)
    gc_2 = keras.layers.Lambda(lambda x: x[0]/x[1])([avg_pool_2 ,sqm_2])

    avg_pool_3 = keras.layers.AveragePooling2D(pool_size=(1,2),strides=(1,2),padding='same')(x_3)
    sq_3 = keras.layers.Multiply()([avg_pool_3,avg_pool_3])
    sqm_3 = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x))(sq_3)
    gc_3 = keras.layers.Lambda(lambda x: x[0]/x[1])([avg_pool_3 ,sqm_3])

    avg_pool_4 = keras.layers.AveragePooling2D(pool_size=(1,2),strides=(1,2),padding='same')(x_4)
    sq_4 = keras.layers.Multiply()([avg_pool_4,avg_pool_4])
    sqm_4 = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x))(sq_4)
    gc_4 = keras.layers.Lambda(lambda x: x[0]/x[1])([avg_pool_4 ,sqm_4])

    sq_5= keras.layers.Multiply()([x_5,x_5])
    sqm_5= keras.layers.Lambda(lambda x: tf.math.reduce_mean(x))(sq_5)
    gc_5 = keras.layers.Lambda(lambda x: x[0]/x[1])([x_5, sqm_5])
    gc_concat = keras.layers.Concatenate(axis=3)([gc_1, gc_2, gc_3, gc_4,gc_5])

    gc_concat = keras.layers.Conv2D(NUM_CLASS,kernel_size=(1,1),strides=(1,1),padding='same',name='conv_out') (gc_concat)
    logits = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x[0],axis=1))([gc_concat])
    output = keras.layers.Softmax()(logits)

    model = keras.Model(inputs =input_layer, outputs = output )
    return model
#%%
