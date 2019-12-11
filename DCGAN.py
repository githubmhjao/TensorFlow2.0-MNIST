# TensorFlow2.0 DCGAN Implementation
# Large amount of credit goes to:
# https://github.com/ChengBinJin/VanillaGAN-TensorFlow
# https://www.tensorflow.org/tutorials/generative/dcgan
# which I've used as a reference for this implementation


import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import time


# ------------------ #
#  Basic Parameters  #
# ------------------ #

Img_H, Img_W = 28, 28

Buffer_Size = 60000
Batch_Size = 100

Latent_Dim = 10

num_examples_to_generate = 16
Seed = tf.random.normal([num_examples_to_generate, Latent_Dim])


# ----------- #
#  Load Data  #
# ----------- #

(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()


# ------------ #
#  Preprocess  #
# ------------ #

def preprocess_data(train_images):
    
    copy_images = np.array(train_images)
    
    copy_images = train_images.reshape(train_images.shape[0], Img_W, Img_H, 1).astype('float32')
    
    # Normalizing the images to the range of [0., 1.]
    copy_images /= 255.
    
    # Binarization
    copy_images[copy_images >= .5] = 1.
    copy_images[copy_images < .5] = 0.
    
    # Create tf Dataset
    copy_dataset = tf.data.Dataset.from_tensor_slices(copy_images).shuffle(Buffer_Size).batch(Batch_Size)
    
    return copy_images, copy_dataset
    
    
train_images, train_dataset = preprocess_data(train_images)


plt.imshow(np.array(train_images[0]).reshape(Img_W, Img_H), cmap=plt.cm.gray)
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

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=2, padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, Img_H, Img_W, 1)

    return model
    

def build_discriminator():
    
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=2, padding='same', input_shape=[Img_H, Img_W, 1]))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=2, padding='same'))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
    
    
g_model = build_generator(Latent_Dim)
d_model = build_discriminator()


_ = g_model(tf.random.normal([1, Latent_Dim]))
plt.imshow(np.array(_).reshape((Img_H, Img_W)))


_ = d_model(tf.random.normal([1, Img_H, Img_W, 1]))
print(_)


# ------------- #
#  Define Loss  #
# ------------- #

# Because there is no activation function on the output of discriminator
# from_logits need to be set as True
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
    
    
# ------------------ #
#  Define Optimizer  #
# ------------------ #

g_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)


# ------------ #
#  Train Step  #
# ------------ #

@tf.function
def train_step(images):
    latent = tf.random.normal([Batch_Size, Latent_Dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = g_model(latent, training=True)

        real_output = d_model(images, training=True)
        fake_output = d_model(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, g_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, d_model.trainable_variables)

    g_optimizer.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
    
    
def sample_save_img(model, epoch, test_input):
    
    record_dir_path = './train_record'

    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    canvas = np.zeros((Img_H * 4, Img_W * 4))

    for i, pred in enumerate(predictions):
        row = i // 4
        col = i % 4
        canvas[row * Img_W:(row + 1) * Img_W, col * Img_W: (col + 1) * Img_H] = np.array(pred).reshape((Img_H, Img_W))

    plt.imshow(canvas, cmap='gray')
    plt.colorbar()
    plt.savefig('{}/image_at_epoch_{:04d}.png'.format(record_dir_path, epoch))
    plt.close()
    
    
def train(dataset, epochs):
    train_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        if (epoch + 1) % 2 == 0:
            sample_save_img(g_model, epoch + 1, Seed)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-epoch_start))

        # Save the model every 15 epochs
        if (epoch + 1) % 1000 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

    # Generate after the final epoch
    sample_save_img(g_model, epochs, Seed)
    print ('Time for train {} is {} sec'.format(epoch + 1, time.time()-train_start))
    
    
# ---------------- #
#  Start Training  #
# ---------------- #

%%time
Epochs = 10
train(train_dataset, Epochs)
