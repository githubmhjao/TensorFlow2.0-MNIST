import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


def down_mbnet_block(input_layer, t, c, s=1):
    """
    :param input_layer: input layer
    :param t: channel expansion factor
    :param c: channel number
    :param s: strides
    
    :return output_layer: output layer
    """
    
    x = layers.Conv2D(t*c, kernel_size=1, strides=1, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    
    x = layers.DepthwiseConv2D(kernel_size=3, strides=s, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    
    x = layers.Conv2D(c, kernel_size=1, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if s != 1:
        short_cut = layers.Conv2D(c, kernel_size=1, strides=s, padding='same')(input_layer)
        short_cut = layers.BatchNormalization()(short_cut)
        
        output_layer = layers.Concatenate()([x, short_cut])
        
    else:
        output_layer = layers.Concatenate()([x, input_layer])
    
    return output_layer


def down_mbnet(input_layer, t, c, s, n):
    """
    :param input_layer: input layer
    :param t: channel expansion factor
    :param c: channel number
    :param s: strides of the first block
    :param n: block number
    
    :return x: output layer
    """
    
    x = down_mbnet_block(input_layer, t, c, s)
    
    for i in range(n-1):
        x = down_mbnet_block(x, t, c, 1)
        
    return x


def encoder_mbnet(input_layer, latent_dim):
    """
    :param input_layer: input layer
    :param latent_dim: latent dimension, int
    
    :return output_layer: output_layer 
    """
    
    x = layers.Dense(1)(input_layer)
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    
    x = down_mbnet(x, 6, 16, 1, 1)
    x = down_mbnet(x, 6, 32, 2, 2)
    x = down_mbnet(x, 6, 64, 1, 1)
    
    x = layers.Conv2D(256, kernel_size=1, strides=1, padding='same')(x)
    x = layers.AveragePooling2D((7, 7))(x)
    x = layers.Conv2D(latent_dim, kernel_size=1, strides=1, padding='same')(x)
    
    output_layer = layers.Flatten()(x)
    
    return output_layer


def up_mbnet_block(input_layer, t, c, s=1):
    """
    :param input_layer: input layer
    :param t: channel expansion factor
    :param c: channel number
    :param s: strides
    
    :return output_layer: output layer
    """
    
    x = layers.Conv2DTranspose(t*c, kernel_size=1, strides=s, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    
    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    
    x = layers.Conv2DTranspose(c, kernel_size=1, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if s != 1:
        short_cut = layers.Conv2DTranspose(c, kernel_size=1, strides=s, padding='same')(input_layer)
        short_cut = layers.BatchNormalization()(short_cut)
        
        output_layer = layers.Concatenate()([x, short_cut])
        
    else:
        output_layer = layers.Concatenate()([x, input_layer])
    
    return output_layer


def up_mbnet(input_layer, t, c, s, n):
    """
    :param input_layer: input layer
    :param t: channel expansion factor
    :param c: channel number
    :param s: strides of the first block
    :param n: block number
    
    :return x: output layer
    """
    
    x = up_mbnet_block(input_layer, t, c, s)
    
    for i in range(n-1):
        x = up_mbnet_block(x, t, c, 1)
        
    return x


def decoder_mbnet(input_layer, latent_dim):
    """
    :param input_layer: input layer
    :param latent_dim: latent dimension, int
    
    :return output_layer: output layer
    """
    
    x = layers.Reshape((1, 1, latent_dim))(input_layer)
    x = layers.UpSampling2D((7, 7))(x)
    
    x = layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    
    x = up_mbnet(x, 6, 64, 1, 1)
    x = up_mbnet(x, 6, 32, 2, 2)
    x = up_mbnet(x, 6, 16, 1, 1)
    
    output_layer = layers.Conv2D(1, kernel_size=1, strides=1, padding='same')(x)
    
    return output_layer
