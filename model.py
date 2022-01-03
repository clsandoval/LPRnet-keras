import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K

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

class small_basic_block(keras.layers.Layer):

    def __init__(self,out_channels,name=None,**kwargs):
        super().__init__(**kwargs)
        out_div4=int(out_channels/4)
        self.main_layers = [
            keras.layers.Conv2D(out_div4,(1,1),padding='same',kernel_initializer=keras.initializers.glorot_uniform(),bias_initializer=keras.initializers.constant()),
            
            keras.layers.ReLU(),
            keras.layers.Conv2D(out_div4,(3,1),padding='same',kernel_initializer=keras.initializers.glorot_uniform(),bias_initializer=keras.initializers.constant()),
            
            keras.layers.ReLU(),
            keras.layers.Conv2D(out_div4,(1,3),padding='same',kernel_initializer=keras.initializers.glorot_uniform(),bias_initializer=keras.initializers.constant()),
            
            keras.layers.ReLU(),
            keras.layers.Conv2D(out_channels,(1,1),padding='same',kernel_initializer=keras.initializers.glorot_uniform(),bias_initializer=keras.initializers.constant()),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ]  
    
    def call(self,input):
        x = input
        for layer in self.main_layers:
            x = layer(x)
        return x

#test this later
class global_context(keras.layers.Layer):
    def __init__(self,kernel_size,stride,**kwargs):
        super().__init__(**kwargs)
        self.ksize = kernel_size
        self.stride = stride


    def call(self, input):
        x = input 
        avg_pool = keras.layers.AveragePooling2D(pool_size=self.ksize,strides=self.stride,padding='same')(x)
        sq = keras.layers.Lambda(lambda x: tf.math.square(x))(avg_pool)
        sqm = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x))(sq)
        out = keras.layers.Lambda(lambda x: tf.math.divide(x[0], x[1]))([avg_pool , sqm])
        return out

class LPRnet(keras.Model):
    def __init__(self, input_shape=(24,94,3), **kwargs):
        super(LPRnet, self).__init__(**kwargs)
        self.input_layer = keras.layers.Input(input_shape)
        self.cnn_layers= [
            keras.layers.Conv2D(64,kernel_size = (3,3),strides=1,padding='same',name='main_conv1',kernel_initializer=keras.initializers.glorot_uniform(),bias_initializer=keras.initializers.constant()),
            keras.layers.BatchNormalization(name='BN1'),
            keras.layers.ReLU(name='RELU1'),
            keras.layers.MaxPool2D(pool_size=(3,3),strides=(1,1),name='maxpool2d_1',padding='same'),
            small_basic_block(128),
            keras.layers.MaxPool2D(pool_size=(3,3),strides=(1,2),name='maxpool2d_2',padding='same'),
            small_basic_block(256),
            small_basic_block(256),
            keras.layers.MaxPool2D(pool_size=(3,3),strides=(1,2),name='maxpool2d_3',padding='same'),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(256,(4,1),strides=1,padding='same',name='main_conv2',kernel_initializer=keras.initializers.glorot_uniform(),bias_initializer=keras.initializers.constant()),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(NUM_CLASS,(1,13),padding='same',name='main_conv3',kernel_initializer=keras.initializers.glorot_uniform(),bias_initializer=keras.initializers.constant()),  
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ]
        self.out_layers = [
            keras.layers.Conv2D(NUM_CLASS,kernel_size=(1,1),strides=(1,1),padding='same',name='conv_out',kernel_initializer=keras.initializers.glorot_uniform(),bias_initializer=keras.initializers.constant()),
        ]
        self.out = self.call(self.input_layer)
        super(LPRnet, self).__init__(
            inputs=self.input_layer,
            outputs=self.out,
            **kwargs)

    def call(self,inputs,training=False):
        x = inputs
        layer_outputs = []
        for layer in self.cnn_layers:
            x = layer(x)
            layer_outputs.append(x)
        scale1 = global_context((1,4),(1,4))(layer_outputs[0]) # first conv
        scale2 = global_context((1,4),(1,4))(layer_outputs[4]) # first small block
        scale3 = global_context((1,2),(1,2))(layer_outputs[7]) # second small block
        sq = keras.layers.Lambda(lambda x: tf.math.square(x))(x)
        sqm = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x))(sq)
        scale4 = keras.layers.Lambda(lambda x: tf.math.divide(x[0], x[1]))([x , sqm])
        gc_concat = keras.layers.Lambda(lambda x: tf.concat([x[0], x[1], x[2], x[3]],3))([scale1, scale2, scale3, scale4])
        for layer in self.out_layers:
            gc_concat = layer(gc_concat)
        logits = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x[0],axis=1))([gc_concat])
        logits = keras.layers.Softmax()(logits)
        return logits