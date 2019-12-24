# TensorFlow2.0 Semi-SupervisedGAN Implementation
# Large amount of credit goes to:
# https://github.com/eriklindernoren/Keras-GAN/tree/master/sgan
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

Img_H, Img_W = 28, 28

Buffer_Size = 60000
Batch_Size = 100

Latent_Dim = 10
Class_Num = 10


# ------------------ #
#  Basic Parameters  #
# ------------------ #

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()



# ------------ #
#  Preprocess  #
# ------------ #

def preprocess_data(train_images, train_labels, buffer=None, batch=None, shuffle=True):
    
    copy_images = np.array(train_images)
    
    copy_images = train_images.reshape(train_images.shape[0], Img_W, Img_H, 1).astype('float32')
    
    # Normalizing the images to the range of [0., 1.]
    copy_images /= 255.
    
    # Binarization
    copy_images[copy_images >= .5] = 1.
    copy_images[copy_images < .5] = 0.
    
    train_labels = train_labels.astype('int32')
    
    # Create tf Dataset
    if shuffle:
        copy_dataset = tf.data.Dataset.from_tensor_slices((copy_images, train_labels)).shuffle(Buffer_Size).batch(Batch_Size)
    else:
        copy_dataset = tf.data.Dataset.from_tensor_slices((copy_images, train_labels)).batch(Batch_Size)
        
    return copy_images, copy_dataset
    
    
train_images, train_dataset = preprocess_data(train_images[:Buffer_Size], train_labels[:Buffer_Size], Buffer_Size, Batch_Size)
test_images, test_dataset = preprocess_data(test_images, test_labels, batch=10000, shuffle=False)

temp_image, temp_num = next(iter(train_dataset))

plt.imshow(np.array(temp_image[0]).reshape((Img_W, Img_H)), cmap=plt.cm.gray)
plt.title(str(temp_num[0].numpy()))
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
    
def build_discriminator(class_num):
    
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
    model.add(layers.Dense(class_num))
    
    return model
    
def custom_activation(output):
    logexpsum = tf.math.reduce_logsumexp(output, 1, keepdims=True)
    activate_output = logexpsum / (logexpsum + 1)
    return activate_output
    
g_model = build_generator(Latent_Dim)
d_model = build_discriminator(Class_Num)

_ = g_model(tf.random.normal([1, Latent_Dim]))
plt.imshow(np.array(_).reshape((Img_H, Img_W)))

_ = d_model(tf.random.normal([10, Img_H, Img_W, 1]))
custom_activation(_)


# ------------- #
#  Define Loss  #
# ------------- #

def cal_gan_loss(target, output):
    # Because we have activated the logits through custom_activation, from_logits has to be set as True
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    if target == 1:
        target_output = tf.ones_like(output)
    elif target == 0:
        target_output = tf.zeros_like(output)
    
    return cross_entropy(target_output, output)
    
def cal_class_loss(real_label, pred_label):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(real_label, pred_label)
    
    
# ------------------ #
#  Define Optimizer  #
# ------------------ #

g_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)


# ---------------- #
#  Define Metrics  #
# ---------------- #

acc = tf.keras.metrics.Accuracy()


# ------------ #
#  Train Step  #
# ------------ #

@tf.function
def train_step(input_image, input_num):
    latent = tf.random.normal([Batch_Size, Latent_Dim])

    with tf.GradientTape() as d_tape:
        g_image = g_model(latent, training=False)
        
        fake_output = d_model(g_image, training = True)
        real_output = d_model(input_image, training=True)
        
        # Calculate GAN loss
        act_fake_output = custom_activation(fake_output)
        act_real_output = custom_activation(real_output)
        gan_loss = cal_gan_loss(0, act_fake_output) + cal_gan_loss(1, act_real_output)
        
        # Calculate classification loss
        c_loss = cal_class_loss(input_num, real_output)
        
        d_loss = gan_loss + c_loss
        
    d_gradient = d_tape.gradient(d_loss, d_model.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradient, d_model.trainable_variables))
        
    with tf.GradientTape() as g_tape:
        g_image = g_model(latent, training=True)

        fake_output = d_model(g_image, training=False)
        
        act_fake_output = custom_activation(fake_output)
        g_loss = cal_gan_loss(1, act_fake_output)

    g_gradient = g_tape.gradient(g_loss, g_model.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradient, g_model.trainable_variables))
    
    
def check_progress(epoch, batch_image, batch_num, test_image, test_num):
    
    record_dir_path = './train_record'

    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    
    # Probe the generator
    seed_image = g_model(Seed, training=False)

    canvas = np.zeros((Img_H * 2, Img_W * 2))

    for i, pred in enumerate(seed_image):
        row = i // 2
        col = i % 2
        canvas[row * Img_W:(row + 1) * Img_W, col * Img_W: (col + 1) * Img_H] = np.array(pred).reshape((Img_H, Img_W))

    plt.imshow(canvas, cmap='gray')
    plt.colorbar()
    plt.savefig('{}/generate_at_epoch_{:04d}.png'.format(record_dir_path, epoch + 1))
    plt.close()
    
    # Probe the classifier
    test_output = d_model(test_image)
    test_output = tf.math.argmax(test_output, axis=1)
    acc.update_state(test_num, test_output)
    print(f'Accuracy at epoch {epoch + 1}:', acc.result().numpy())
    
    for i in range(Class_Num):
        mask = np.array(test_num) == i
        plt.scatter(np.array(test_num)[mask], np.array(test_output)[mask], alpha=0.2)
        
    plt.savefig('{}/classify_at_epoch_{:04d}.png'.format(record_dir_path, epoch + 1))
    plt.close()
    

def train(dataset, epochs):
    train_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()

        for batch_image, batch_num in dataset:
            train_step(batch_image, batch_num)

        # Produce images for the GIF as we go
        if (epoch + 1) % 2 == 0:
            test_image, test_num = next(iter(test_dataset))
            check_progress(epoch, batch_image, batch_num, test_image, test_num)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-epoch_start))


    # Generate after the final epoch
    check_progress(epoch, batch_image, batch_num, test_image, test_num)
    print ('Time for train {} is {} sec'.format(epoch + 1, time.time()-train_start))
    
    
# ---------------- #
#  Start Training  #
# ---------------- #
%%time
Seed = tf.random.normal([4, Latent_Dim])

Epochs = 20
train(train_dataset, Epochs)

# --------------------------------------------- #
#  Accuracy at epoch 2: 0.9                     #
#  Time for epoch 2 is 11.655494213104248 sec   #
#  Accuracy at epoch 4: 0.915                   #
#  Time for epoch 4 is 11.694486856460571 sec   #
#  Accuracy at epoch 6: 0.91                    #
#  Time for epoch 6 is 11.627080202102661 sec   #
#  Accuracy at epoch 8: 0.9175                  #
#  Time for epoch 8 is 11.51382040977478 sec    #
#  Accuracy at epoch 10: 0.93                   #
#  Time for epoch 10 is 11.515087366104126 sec  #
#  Accuracy at epoch 12: 0.93666667             #
#  Time for epoch 12 is 11.538561582565308 sec  #
#  Accuracy at epoch 14: 0.9442857              #
#  Time for epoch 14 is 11.517507076263428 sec  #
#  Accuracy at epoch 16: 0.95                   #
#  Time for epoch 16 is 11.540745258331299 sec  #
#  Accuracy at epoch 18: 0.9533333              #
#  Time for epoch 18 is 11.42297911643982 sec   #
#  Accuracy at epoch 20: 0.957                  #
#  Time for epoch 20 is 11.484888553619385 sec  #
#  Accuracy at epoch 20: 0.96                   #
#  Time for train 20 is 232.55758333206177 sec  #
#  Wall time: 3min 52s                          #
# --------------------------------------------- # 
