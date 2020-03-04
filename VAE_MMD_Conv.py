# TensorFlow2.0 VAE Implementation
# Large amount of credit goes to:
# https://github.com/ChengBinJin/VAE-Tensorflow
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
# https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.py
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

ImgW, ImgH = 28, 28

BufferSize = 60000
BatchSize = 100

LatentDim = 2

Seed = np.random.randint(0, high=BufferSize, size=4)


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
    
    copy_images = train_images.reshape(train_images.shape[0], ImgW, ImgH, 1).astype('float32')
    
    # Normalizing the images to the range of [0., 1.]
    copy_images /= 255.
    
    # Binarization
    copy_images[copy_images >= .5] = 1.
    copy_images[copy_images < .5] = 0.
    
    # Create tf Dataset
    copy_dataset = tf.data.Dataset.from_tensor_slices(copy_images).shuffle(BufferSize).batch(BatchSize)
    
    return copy_images, copy_dataset
    
    
train_images, train_dataset = preprocess_data(train_images)
noise_images, noise_dataset = preprocess_data(noise_images)

plt.imshow(np.array(noise_images[0]).reshape((ImgW, ImgH)), cmap=plt.cm.gray)
plt.colorbar()

plt.imshow(np.array(train_images[0]).reshape(ImgW, ImgH), cmap=plt.cm.gray)
plt.colorbar()


# ------------- #
#  Build Model  #
# ------------- #

def build_encoder(latent_dim):
    
    i = layers.Input(shape=(ImgW, ImgH, 1))
    d = layers.Dense(1)(i)
    
    # Not able to connect Input with Conv2D, why?
    x = layers.Conv2D(64,(3, 3), padding='valid',activation='relu')(d)
    x = layers.Conv2D(64,(3, 3), padding='valid',activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32,(3, 3), padding='valid',activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    
    z = layers.Dense(latent_dim)(x)
    
    return Model(i, z, name='encoder')


def build_decoder(latent_dim):
    
    z = layers.Input(shape=latent_dim)
    
    x = layers.Reshape([1,1,latent_dim])(z)
    
    x = layers.Conv2DTranspose(128,(1, 1), padding='valid',activation='relu')(x)
    x = layers.Conv2DTranspose(64,(3, 3), padding='valid',activation='relu')(x)
    x = layers.Conv2DTranspose(64,(3, 3), padding='valid',activation='relu')(x)
    x = layers.Conv2DTranspose(48,(3, 3), strides=(2, 2),padding='same',activation='relu')(x)
    x = layers.Conv2DTranspose(48,(3, 3), padding='valid',activation='relu')(x)
    x = layers.Conv2DTranspose(32,(3, 3), strides=(2, 2),padding='same',activation='relu')(x)
    x = layers.Conv2DTranspose(16,(3, 3), padding='valid',activation='relu')(x)

    output = layers.Conv2DTranspose(1,(3, 3), padding='valid',activation='sigmoid')(x)
    
    return Model(z, output, name='decoder')
    

encoder = build_encoder(LatentDim)
decoder = build_decoder(LatentDim)


encoder(tf.random.normal([1, ImgW, ImgH, 1]))

_ = decoder(tf.random.normal([1, LatentDim]))
plt.imshow(np.array(_).reshape((ImgW, ImgH)), cmap=plt.cm.gray)
plt.colorbar()


# ------------- #
#  Define Loss  #
# ------------- #

def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

def cal_m_loss(z):
    assert z.shape[0] == BatchSize
    assert z.shape[1] == LatentDim
    true_samples = tf.random.normal(tf.shape(z), dtype=tf.float32)
    loss_mmd = compute_mmd(true_samples, z)
    return loss_mmd
    
def cal_r_loss(img_input, img_output):
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
    
    # Tune the reconstruction level??
    r_to_m_loss_ratio = 0.1
    
    with tf.GradientTape() as tape:
        z = encoder(img_input)

        img_output = decoder(z)

        loss = cal_r_loss(img_input, img_output) * ImgW * ImgH * r_to_m_loss_ratio + cal_m_loss(z)
    
    vae_param = encoder.trainable_variables + decoder.trainable_variables
    
    gradients_of_vae = tape.gradient(loss, vae_param)
    optimizer.apply_gradients(zip(gradients_of_vae, vae_param))
    
    
def latent_map(images, labels):
    
    for i in range(10):
        z = encoder(images[labels == i], training=False)
    
        z_distribution = [[], []]
        for j in np.array(z):
            z_distribution[0].append(j[0])
            z_distribution[1].append(j[1])

        plt.scatter(z_distribution[0], z_distribution[1], alpha=0.8, label=i)
        
    plt.legend()
    
    return plt.gcf()
    

def sample_save_img(seed, images, epoch, use_VAE=False):
    
    # Please create a dir './train_record'
    record_dir_path = './train_record'
    
    # Probe the decoder progress
    fig, axes = plt.subplots(2, 2)
    
    for i, ax in zip(seed, axes.flatten()):
        
        if use_VAE:
            z = encoder(images[i].reshape(1, ImgW, ImgH, 1))
            sample = decoder(z)
            
        else:
            sample = images[i]
        
        im = ax.imshow(np.array(sample).reshape(ImgW, ImgH), cmap=plt.cm.gray)
    
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
        
        if (epoch + 1) % 2 == 0:
            sample_save_img(Seed, images, epoch, use_VAE=True)
            print ('Time for epoch {} is {:.2f} sec'.format(epoch + 1, time.time()-epoch_start))
            
    # Generate after the final epoch
    sample_save_img(Seed, images, epoch)
    print('='*30)
    print ('Time for train {} epochs is {:.2f} sec'.format(epoch + 1, time.time()-train_start))


# ---------------- #
#  Start Training  #
# ---------------- #

Epochs = 30
train(noise_images, noise_dataset, Epochs)

# -----------------------------------------#
#  Training Record                         #
#                                          #
#  Time for epoch 2 is 6.78 sec            #
#  Time for epoch 4 is 6.07 sec            #
#  Time for epoch 6 is 5.95 sec            #
#  Time for epoch 8 is 6.01 sec            #
#  Time for epoch 10 is 5.96 sec           #
#  Time for epoch 12 is 6.19 sec           #
#  Time for epoch 14 is 5.99 sec           #
#  Time for epoch 16 is 6.16 sec           #
#  Time for epoch 18 is 6.11 sec           #
#  Time for epoch 20 is 6.00 sec           #
#  Time for epoch 22 is 6.02 sec           #
#  Time for epoch 24 is 6.00 sec           #
#  Time for epoch 26 is 6.26 sec           #
#  Time for epoch 28 is 6.12 sec           #
#  Time for epoch 30 is 5.95 sec           #
#  ==============================          #
#  Time for train 30 epochs is 178.98 sec  #
# ---------------------------------------- #


def image_map(x_min, x_max, y_min, y_max, x_dim=0, y_dim=1):
    
    record_dir_path = './train_record'
    
    fix_param = np.random.rand(LatentDim)

    fig, axes = plt.subplots(20, 20, figsize=(10, 10))
    fig.subplots_adjust(hspace=0, wspace=0)
    
    count = 0
    x_step = (x_max - x_min) / 20
    y_step = (y_max - y_min) / 20
    
    for i in np.arange(y_min, y_max, y_step):
        for j in np.arange(x_min, x_max, x_step):
            _ = list(fix_param)
            _[x_dim] = i
            _[y_dim] = j
            _ = decoder(np.array(_).reshape(1, LatentDim), training=False)
            axes.ravel()[count].imshow(np.array(_).reshape(ImgW, ImgH), cmap=plt.cm.gray)
            axes.ravel()[count].axis('off')
            axes.ravel()[count].axis('tight')
            count += 1
            
    fig.savefig(f'{record_dir_path}/image_map.png')
    
    
image_map(-1, 1, -1, 1)
