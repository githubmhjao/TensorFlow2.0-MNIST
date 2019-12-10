# TensorFlow2.0 VAE Implementation
# Large amount of credit goes to:
# https://github.com/ChengBinJin/VAE-Tensorflow
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
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

Img_W, Img_H = 28, 28

Buffer_Size = 60000
Batch_Size = 100

Latent_Dim = 10

Seed = np.random.randint(0, high=Buffer_Size, size=4)


# ----------- #
#  Load Data  #
# ----------- #

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

noise_images = train_images + 128 * np.random.randn(*train_images.shape)
noise_images = np.clip(noise_images, 0., 255.)


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
noise_images, noise_dataset = preprocess_data(noise_images)

plt.imshow(np.array(noise_images[0]).reshape((Img_W, Img_H)), cmap=plt.cm.gray)
plt.colorbar()

plt.imshow(np.array(train_images[0]).reshape(Img_W, Img_H), cmap=plt.cm.gray)
plt.colorbar()


# ------------- #
#  Build Model  #
# ------------- #

def build_encoder(nodes, latent_dim):
    
    i = layers.Input(shape=(Img_W, Img_H, 1))

    x = layers.Flatten()(i)
    x = layers.Dense(nodes, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(nodes, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    z_mean = layers.Dense(latent_dim)(x)
    z_logvar = layers.Dense(latent_dim)(x)
    
    return Model(i, [z_mean, z_logvar], name='encoder')
    
def z_sample(z_mean, z_logvar):
    eps = tf.random.normal(tf.shape(z_mean), mean=0., stddev=1., dtype=tf.float32)
    z = z_mean + tf.exp(0.5 * z_logvar) * eps
    return z
    
def build_decoder(nodes, latent_dim):
    
    z = layers.Input(shape=latent_dim)
    
    x = layers.Dense(nodes, activation='relu')(z)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(nodes, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    output = layers.Dense(Img_W * Img_H * 1, activation='sigmoid')(x)
    output = layers.Reshape((Img_W, Img_H, 1))(output)
    
    return Model(z, output, name='decoder')
    
encoder = build_encoder(500, Latent_Dim)
decoder = build_decoder(500, Latent_Dim)

encoder(tf.random.normal([1, Img_W, Img_H, 1]))

_ = decoder(tf.random.normal([1, Latent_Dim]))
plt.imshow(np.array(_).reshape((Img_W, Img_H)))


# ------------- #
#  Define Loss  #
# ------------- #

def z_loss(mean, logvar):
    kl_divergence = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), 1)
    return tf.reduce_mean(kl_divergence)
    
def reconstruction_loss(img_input, img_output):
    # Because we have applied 'sigmoid' to our img_output in neural network
    # Therefore, 'from_logits' has to be set as False, which means img_output is not a raw probability
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return cross_entropy(img_input, img_output)


# ------------------ #
#  Define Optimizer  #
# ------------------ #

optimizer = tf.keras.optimizers.Adam(1e-4)


# ------------ #
#  Train Step  #
# ------------ #

@tf.function
def train_step(img_input):
    
    # Tune the reconstruction level
    weight = 0.5
    
    with tf.GradientTape() as tape:
        z_mean, z_logvar = encoder(img_input)

        # Sampling
        z = z_sample(z_mean, z_logvar)

        img_output = decoder(z)

        loss = reconstruction_loss(img_input, img_output) * Img_W * Img_H * weight + z_loss(z_mean, z_logvar)
    
    vae_param = encoder.trainable_variables + decoder.trainable_variables
    
    gradients_of_vae = tape.gradient(loss, vae_param)
    optimizer.apply_gradients(zip(gradients_of_vae, vae_param))
    
def latent_map(images, labels):
    
    for i in range(10):
        z_mean, z_logvar = encoder(images[labels == i], training=False)
        z = z_sample(z_mean, z_logvar)
    
        z_distribution = [[], []]
        for j in np.array(z):
            z_distribution[0].append(j[0])
            z_distribution[1].append(j[1])

        plt.scatter(z_distribution[0], z_distribution[1], alpha=0.8, label=i)
        
    plt.legend()
    
    return plt.gcf()
    
def sample_save_img(seed, images, epoch, use_VAE=False):
    
    record_dir_path = './train_record'
    
    # Probe the decoder progress
    fig, axes = plt.subplots(2, 2)
    
    for i, ax in zip(seed, axes.flatten()):
        
        if use_VAE:
            z_mean, z_logvar = encoder(images[i].reshape(1, Img_W, Img_H, 1))
            z = z_sample(z_mean, z_logvar)
            sample = decoder(z)
            
        else:
            sample = images[i]
        
        im = ax.imshow(np.array(sample).reshape(Img_W, Img_H), cmap=plt.cm.gray)
    
    fig.colorbar(im, ax=axes.ravel().tolist())
    
    if use_VAE:
        plt.savefig('{}/image_at_epoch_{:04d}.png'.format(record_dir_path, epoch + 1))
    else:
        plt.savefig('{}/ground_truth.png'.format(record_dir_path))
    plt.close()
    
    if use_VAE:
        # Probe the encoder progress
        _ = latent_map(images[:1000], train_labels[:1000])

        plt.savefig('{}/distri_at_epoch_{:04d}.png'.format(record_dir_path, epoch + 1))
        plt.close()
        
def train(images, dataset, epochs):
    
    train_start = time.time()
    
    for epoch in range(epochs):
        
        epoch_start = time.time()
        
        for batch in dataset:
            train_step(batch)
        
        if (epoch + 1) % 5 == 0:
            sample_save_img(Seed, images, epoch, use_VAE=True)
            print ('Time for epoch {} is {:.2f} sec'.format(epoch + 1, time.time()-epoch_start))
            
    # Generate after the final epoch
    sample_save_img(Seed, images, epoch)
    print('='*30)
    print ('Time for train {} epochs is {:.2f} sec'.format(epoch + 1, time.time()-train_start))
    

# ---------------- #
#  Start Training  #
# ---------------- #

Epochs = 20
train(noise_images, noise_dataset, Epochs)

# --------------------------------------- #
#  Time for epoch 5 is 1.44 sec           #
#  Time for epoch 10 is 1.41 sec          #
#  Time for epoch 15 is 1.72 sec          #
#  Time for epoch 20 is 1.40 sec          #
#  ==============================         #
#  Time for train 20 epochs is 24.68 sec  #
# --------------------------------------- #

def image_map(x_dim=0, y_dim=1):
    
    record_dir_path = './train_record'
    
    fix_param = np.random.rand(10)

    fig, axes = plt.subplots(20, 20, figsize=(10, 10))
    fig.subplots_adjust(hspace=0, wspace=0)
    
    count = 0
    for i in range(-20, 20, 2):
        for j in range(-20, 20, 2):
            _ = list(fix_param)
            _[x_dim] = i/10
            _[y_dim] = j/10
            _ = decoder(np.array(_).reshape(1, 10), training=False)
            axes.ravel()[count].imshow(np.array(_).reshape(Img_W, Img_H), cmap=plt.cm.gray)
            axes.ravel()[count].axis('off')
            axes.ravel()[count].axis('tight')
            count += 1
            
    fig.savefig(f'{record_dir_path}/image_map.png')
    
image_map()
