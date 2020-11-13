import tensorflow as tf
from VAE_module.models import VAE
from VAE_module.callbacks import ReduceReconstructionLoss
from VAE_module.layers import Sampling
import numpy as np


def make_prior(x_dim, latent_dim):
    input_x = tf.keras.Input(x_dim)
    x = tf.keras.layers.Dense(32, activation="relu")(input_x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    z_prior = tf.keras.Model(input_x, [z_mean, z_log_var, z], name="z_prior")
    return z_prior

def make_encoder(x_dim, y_dim, latent_dim):
    input_x = tf.keras.Input(x_dim)
    input_y = tf.keras.Input(y_dim)
    x = tf.keras.layers.concatenate([input_x, input_y])
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model([input_x, input_y], [z_mean, z_log_var, z], name="encoder")
    return encoder

def make_decoder(x_dim, y_dim, latent_dim):
    input_x = tf.keras.Input(shape=(x_dim))
    input_z = tf.keras.Input(shape=(latent_dim))

    x = tf.keras.layers.concatenate([input_x, input_z])
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    y_mean = tf.keras.layers.Dense(y_dim, name="x_mean")(x)
    y_log_var = tf.keras.layers.Dense(y_dim, name="x_log_var")(x)
    y = Sampling()([y_mean, y_log_var])
    decoder = tf.keras.Model([input_x, input_z], [y_mean, y_log_var, y], name="decoder")
    return decoder


if __name__ == "__main__":
    data_x = np.random.randint(0,10, [1000,1])/10
    data_y = np.random.normal(data_x, np.ones(shape=data_x.shape)*0.01)

    encoder = make_encoder(1,1,2)
    decoder = make_decoder(1,1,2)
    prior = make_prior(1,2)

    vae = VAE(encoder=encoder, prior=prior, decoder=decoder, reconstruction_weight=1)
    vae.compile(optimizer="adam")
    vae.fit([data_x, data_y], epochs=100, callbacks=ReduceReconstructionLoss(method="exponential"))

    prediction = vae.predict([data_x.repeat(100, axis=0)])
