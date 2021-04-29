

import Load_Datasets

from tensorflow.keras import layers
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, LeakyReLU
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16

from IPython import display

# this is the width and height of the images
inputsize = 32

path = r'?\WGANAutofocus' # Replace ? with location of WGANAutofocus


def perceptual_loss(
    y_true,
    y_pred,
    loss_model=None,
    sample_weight=None,
    input_shape=(inputsize, inputsize, 3),
    loss_factor=1,
):

    if loss_model is None:
        vgg = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
        loss_model = Model(
            inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output
        )
        loss_model.trainable = False
    return loss_factor * K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def evaluate_psnr(model, dataset, evaluation_steps=10):
    return tf.math.reduce_mean(
        [
            tf.image.psnr(org, model(cor), max_val=1)
            for cor, org in dataset.take(evaluation_steps)
        ]
    )










def Image_Gen():
    Input_img = Input(shape=(inputsize, inputsize, 3))

    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(Input_img)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x1)
    x3 = MaxPool2D(padding='same')(x2)
    x4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x3)
    x5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x4)
    x6 = MaxPool2D(padding='same')(x5)
    encoded = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x6)



    x7 = UpSampling2D()(encoded)


    x8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x7)
    x9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x8)
    x10 = Add()([x5, x9])
    x11 = UpSampling2D()(x10)
    x12 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x11)
    x13 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x12)
    x14 = Add()([x2, x13])
    x15 = LeakyReLU()(x14)
    decoded = Conv2D(3, (3, 3), padding='same', activation=tf.nn.tanh, kernel_regularizer=regularizers.l1(10e-10))(x15)


    Generator = Model(Input_img, decoded)
    Generator.summary()
    return Generator

channel_rate = 64
ngf = 64
ndf = 64
input_nc = 3
output_nc = 3
n_blocks_gen = 9







def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[inputsize, inputsize, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    n_layers, use_sigmoid = 3, False
    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
        model.add(layers.Conv2D(
            filters=64 * nf_mult, kernel_size=(4, 4), strides=2, padding="same"
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(Dense(1, activation="sigmoid"))

    model.summary()

    return model










def train_step(Cor_batch, Org_batch, loss_model, Image_Generator, discriminator, generator_optimizer, discriminator_optimizer):



    for i in range(4):
        with tf.GradientTape() as tape:

          generated_images = Image_Generator(Cor_batch, training=True)

          real_output = discriminator(Org_batch, training=True)
          d_loss_real = wasserstein_loss(
              tf.ones_like(real_output), real_output
          )
          fake_output = discriminator(generated_images, training=True)
          d_loss_fake = wasserstein_loss(
              -tf.ones_like(fake_output), fake_output
          )


          disc_loss = tf.math.reduce_mean(0.5 * tf.math.add(d_loss_real, d_loss_fake))


        gradients_of_discriminator = tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



    with tf.GradientTape() as tape:
        deblurred_images = Image_Generator(Cor_batch)
        predictions = discriminator(deblurred_images)


        discriminator_loss = wasserstein_loss(tf.ones_like(predictions), predictions)

        image_loss = perceptual_loss(
            Org_batch, deblurred_images, loss_model=loss_model
        )
        mse = (tf.keras.losses.MSE((Org_batch), Image_Generator(Cor_batch)))



        g_loss = tf.math.reduce_mean(10*image_loss + discriminator_loss + mse)

    gradients = tape.gradient(g_loss, Image_Generator.trainable_weights)
    generator_optimizer.apply_gradients(zip(gradients, Image_Generator.trainable_weights))

    return g_loss, disc_loss, mse

def train(train_dataset,test_dataset, epochs, loss_model, steps_per_epoch, Image_Generator, discriminator, generator_optimizer, discriminator_optimizer):


  for epoch in range(epochs):
    print("epoch: %d", epoch)
    progbar = tf.keras.utils.Progbar(steps_per_epoch)


    for step_index, (Org_batch, Cor_batch) in enumerate(train_dataset):
        g_loss, d_loss, mse = train_step(Cor_batch, Org_batch, loss_model, Image_Generator, discriminator, generator_optimizer, discriminator_optimizer)


        progbar.update(step_index, values=[("d_loss", d_loss), ("g_loss", g_loss), ("mse", mse)])



    Image_Generator.save_weights("Random.h5")

    for step_index, (test_Org_batch, test_Cor_batch) in enumerate(test_dataset):
        generate_and_save_images(Image_Generator,
                                 epoch + 1,
                                 test_Org_batch,
                                 test_Cor_batch)

    display.clear_output(wait=True)





def generate_and_save_images(model, epoch, data_org, data_cor):
    predictions = (model(data_cor, training=False)).numpy()

    test_org = np.stack(list(data_org))
    test_cor = np.stack(list(data_cor))
    for i in range(4):
        plt.subplot(4, 4, i + 1)
        plt.imshow((test_org[i, :, :, 0] * 127.5 + 127.5), cmap='gray')
        plt.axis('off')
    for i in range(4):
        plt.subplot(4, 4, i + 5)
        plt.imshow((predictions[i, :, :, 0] * 127.5 + 127.5), cmap='gray')
        plt.axis('off')
    for i in range(4):
        plt.subplot(4, 4, i + 9)
        plt.imshow((test_cor[i, :, :, 0] * 127.5 + 127.5), cmap='gray')
        plt.axis('off')

    plt.savefig(path + "Images/image_at_epoch_{:04d}.png".format(epoch))
    plt.close()


def run():


    Image_Generator = Image_Gen()




    discriminator = make_discriminator_model()
    discriminator.summary()



    patch_size = (inputsize,inputsize)
    vgg = tf.keras.applications.VGG16(
            include_top=False, weights="imagenet", input_shape=(*patch_size, 3)
        )
    loss_model = tf.keras.Model(
            inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output
        )
    loss_model.trainable = False





    batch_size = 32
    patch_size = (inputsize,inputsize)
    test_dataset, test_dataset_length = Load_Datasets.load(
        "Random",
        mode="Test",
        batch_size=batch_size,
        patch_size=patch_size,
        shuffle=False,
        path=path,
        size=inputsize
    )

    batch_size = 32
    patch_size = (inputsize,inputsize)
    train_dataset, train_dataset_length = Load_Datasets.load(
        "Random",
        mode="Train",
        batch_size=batch_size,
        patch_size=patch_size,
        shuffle=False,
        path=path,
        size=inputsize
    )


    steps_per_epoch = train_dataset_length // batch_size


    generator_optimizer = tf.keras.optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr=3e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)



    EPOCHS = 100



    train(train_dataset,test_dataset, EPOCHS, loss_model, steps_per_epoch, Image_Generator, discriminator, generator_optimizer, discriminator_optimizer)


if __name__ == '__main__':
    run()

