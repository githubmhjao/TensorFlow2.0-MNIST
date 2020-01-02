# TensorFlow2.0 Energy-BasedGAN Implementation
# Large amount of credit goes to:
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/ebgan/ebgan.py
# which I've used as a reference for this implementation


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

import numpy as np
import matplotlib.pyplot as plt
import time


# ------------------ #
#  Basic Parameters  #
# ------------------ #

imgH, imgW = 28, 28

bufferSize = 60000
batchSize = 100

latentDim = 20


# ----------- #
#  Load Data  #
# ----------- #

(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()


# ------------ #
#  Preprocess  #
# ------------ #

def preprocess_data(train_images):
    
    copy_images = np.array(train_images)
    
    copy_images = train_images.reshape(train_images.shape[0], imgH, imgW, 1).astype('float32')
    
    # Normalizing the images to the range of [0., 1.]
    copy_images /= 255.
    
    # Binarization
    copy_images[copy_images >= .5] = 1.
    copy_images[copy_images < .5] = 0.
    
    # Create tf Dataset
    copy_dataset = tf.data.Dataset.from_tensor_slices((copy_images)).shuffle(bufferSize).batch(batchSize)
        
    return copy_images, copy_dataset
    
    
train_images, train_dataset = preprocess_data(train_images)

temp_image = next(iter(train_dataset))

plt.imshow(np.array(temp_image[0]).reshape((imgH, imgW)), cmap=plt.cm.gray)
plt.colorbar()


# ------------- #
#  Build Model  #
# ------------- #

def build_generator(latent_dim):

    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (3, 3), strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (3, 3), strides=2, padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, imgH, imgW, 1)

    return model
    
    
def build_discriminator(latent_dim):
    
    encode = tf.keras.Sequential()
    encode.add(layers.Conv2D(64, (3, 3), strides=2, padding='same', input_shape=[imgH, imgW, 1]))
    assert encode.output_shape == (None, 14, 14, 64)
    encode.add(layers.LeakyReLU())
    encode.add(layers.Dropout(0.3))

    encode.add(layers.Conv2D(128, (3, 3), strides=2, padding='same'))
    assert encode.output_shape == (None, 7, 7, 128)
    encode.add(layers.LeakyReLU())
    encode.add(layers.Dropout(0.3))
    
    encode.add(layers.Flatten())
    encode.add(layers.Dense(latent_dim))
    
    
    decode = tf.keras.Sequential()
    decode.add(layers.Dense(7*7*128, input_shape=[latent_dim]))
    decode.add(layers.Reshape([7, 7, 128]))
    decode.add(layers.Conv2DTranspose(128, (3, 3), strides=1, padding='same'))
    assert decode.output_shape == (None, 7, 7, 128)
    decode.add(layers.LeakyReLU())
    decode.add(layers.Dropout(0.3))
    
    decode.add(layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same'))
    assert decode.output_shape == (None, 14, 14, 64)
    decode.add(layers.LeakyReLU())
    decode.add(layers.Dropout(0.3))
    
    decode.add(layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same'))
    assert decode.output_shape == (None, 28, 28, 32)
    decode.add(layers.LeakyReLU())
    decode.add(layers.Dropout(0.3))
    
    decode.add(layers.Conv2DTranspose(1, (1, 1), strides=1, activation='sigmoid'))
    assert decode.output_shape == (None, 28, 28, 1)
    
    input_image = layers.Input([imgH, imgW, 1])
    embeddings = encode(input_image)
    
    output_image = decode(embeddings)
    
    return Model(input_image, [embeddings, output_image])
    
    
g_model = build_generator(latentDim)
d_model = build_discriminator(latentDim)


temp_g_img = g_model(tf.random.normal([10, latentDim]))
plt.imshow(np.array(temp_g_img[0]).reshape(imgH, imgW), cmap='gray')


temp_e_output, temp_d_output = d_model(temp_g_img)
plt.imshow(np.array(temp_d_output[0]).reshape(imgH, imgW), cmap='gray')


# ------------- #
#  Define Loss  #
# ------------- #

def cal_pt_loss(embeddings):
    
    norm = tf.sqrt(tf.math.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    similarity = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
    batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
    pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    
    return pt_loss
    
    
def cal_gan_loss(input_image, decode_image):
    loss = tf.reduce_mean(tf.keras.losses.MSE(input_image, decode_image))
    return loss
    

# ------------------ #
#  Define Optimizer  #
# ------------------ #

g_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)


# ---------------- #
#  Define Metrics  #
# ---------------- #

fake_loss_metrics = tf.keras.metrics.Mean()
real_loss_metrics = tf.keras.metrics.Mean()


# ------------ #
#  Train Step  #
# ------------ #

@tf.function
def train_step(input_image, margin):
    
    # pt_weight = 0.1
    latent = tf.random.normal([batchSize, latentDim])

    with tf.GradientTape() as d_tape:
        g_image = g_model(latent, training=False)
        
        _, fake_output = d_model(g_image, training = True)
        _, real_output = d_model(input_image, training=True)
        
        # Calculate GAN loss
        fake_loss = cal_gan_loss(g_image, fake_output)
        real_loss = cal_gan_loss(input_image, real_output)
        
        d_loss = tf.math.maximum((margin - fake_loss), 0) + real_loss
        
    d_gradient = d_tape.gradient(d_loss, d_model.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradient, d_model.trainable_variables))
    
    fake_loss_metrics(fake_loss)
    real_loss_metrics(real_loss)
        
    with tf.GradientTape() as g_tape:
        g_image = g_model(latent, training=True)

        fake_embeddings, fake_output = d_model(g_image, training=False)
        
        # Calculate GAN loss
        g_loss = cal_gan_loss(g_image, fake_output)
        # g_loss += pt_weight * cal_pt_loss(fake_embeddings)

    g_gradient = g_tape.gradient(g_loss, g_model.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradient, g_model.trainable_variables))
    
    
    
def check_progress(epoch, batch_image):
    
    record_dir_path = './train_record'

    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    
    # Probe the generator
    seed_image = g_model(seedLatent, training=False)

    canvas = np.zeros((imgH * 4, imgW * 4))

    for i, pred in enumerate(seed_image):
        row = i // 4
        col = i % 4
        canvas[row * imgH:(row + 1) * imgH, col * imgW: (col + 1) * imgW] = np.array(pred).reshape((imgH, imgW))

    plt.imshow(canvas, cmap='gray')
    plt.colorbar()
    plt.savefig('{}/generator_at_epoch_{:04d}.png'.format(record_dir_path, epoch + 1))
    plt.close()
    
    # Probe the decoder
    canvas = np.zeros((imgH * 4, imgW * 4))
    
    _, seed_image = d_model(seedImage)
    
    for i, pred in enumerate(seed_image):
        row = i // 4
        col = i % 4
        canvas[row * imgH:(row + 1) * imgH, col * imgW: (col + 1) * imgW] = np.array(pred).reshape((imgH, imgW))

    plt.imshow(canvas, cmap='gray')
    plt.colorbar()
    plt.savefig('{}/discriminator_at_epoch_{:04d}.png'.format(record_dir_path, epoch + 1))
    plt.close()
    
    
    
def train(dataset, epochs, margin):
    train_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()

        for batch_image in dataset:
            train_step(batch_image, margin)

        # Produce images for the GIF as we go
        if (epoch + 1) % 2 == 0:
            check_progress(epoch, batch_image)
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-epoch_start))
            print('generator loss:', fake_loss_metrics.result().numpy())
            print('discriminator loss:', real_loss_metrics.result().numpy())
            
            fake_loss_metrics.reset_states()
            real_loss_metrics.reset_states()

    # Generate after the final epoch
    check_progress(epoch, batch_image)
    print ('Time for train {} is {} sec'.format(epoch + 1, time.time()-train_start))
    
    
# ---------------- #
#  Start Training  #
# ---------------- #

%%time
seedLatent = tf.random.normal([16, latentDim])
seedImage = next(iter(train_dataset))[:16]

Epochs = 20
margin = 0.1
train(train_dataset, Epochs, margin)
