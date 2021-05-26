import tensorflow as tf

from variational_autoencoder.models import VAE
from variational_autoencoder.layers import Sampling
from variational_autoencoder.layers import StackNTimes
from variational_autoencoder.losses import KLDivergence
from variational_autoencoder.callbacks import BetaScaling


class savefig(tf.keras.callbacks.Callback):
    def __init__(self, location="figures", **kwargs):
        super(savefig, self).__init__()
        self.location = location
    def on_epoch_begin(self, epoch, logs=None):
        nsmpl = 15
        res = vae.predict(tf.one_hot(np.linspace(0, 9, 10).repeat(nsmpl), 10)).reshape(10, nsmpl, 28, 28).transpose(
            [1, 0, 2, 3])
        fig, axes = plt.subplots(nrows=nsmpl, ncols=10, figsize=(10, int(nsmpl / 2)))

        for i in range(nsmpl):
            for j in range(10):
                axes[i, j].imshow(res[i, j], cmap="Greys")
                axes[i, j].set_axis_off()

        # fig.show()
        fig.savefig("{}/mnist_{}.png".format(self.location, epoch))


def make_prior(latent_dim=2):
    # Sample input for compatibility, ignored in MNIST example
    samples = tf.keras.Input(shape=[], batch_size=1)
    
    prior_input = tf.keras.Input(shape=10)
    x = tf.keras.layers.Dense(64, activation="relu")(prior_input)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z_params = tf.stack([z_mean, z_log_var], axis=-1)
    z = Sampling()([z_mean, tf.exp(z_log_var/2)])
    return tf.keras.Model([prior_input, samples], [z_params, z], name="prior")

def make_encoder(latent_dim=2):
    # Sample input for compatibility, ignored in MNIST example
    samples = tf.keras.Input(shape=[], batch_size=1)
    
    encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)

    label_input = tf.keras.Input(10)
    x = tf.keras.layers.concatenate([label_input, x])
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z_params = tf.stack([z_mean, z_log_var], axis=-1)
    z = Sampling()([z_mean, tf.exp(z_log_var/2)])
    return tf.keras.Model([label_input, encoder_inputs, samples], [z_params, z], name="encoder")

def make_decoder(latent_dim=2):
    # Sample input for compatibility, ignored in MNIST example
    samples = tf.keras.Input(shape=[], batch_size=1)
    
    latent_inputs = tf.keras.Input(shape=latent_dim)
    label_inputs = tf.keras.Input(shape=10)

    x = tf.keras.layers.concatenate([label_inputs, latent_inputs])
    x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu")(x)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    y = tf.keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    return tf.keras.Model([label_inputs, latent_inputs, samples], [y, y], name="decoder")



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    mnist_labels = np.concatenate([y_train, y_test], axis=0)
    mnist_labels_one_hot = tf.one_hot(mnist_labels, depth=10).numpy()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255


    latent_dim = 10

    prior = make_prior(latent_dim)
    encoder = make_encoder(latent_dim)
    decoder = make_decoder(latent_dim)

    vae = VAE(encoder=encoder, decoder=decoder, prior=prior, 
              inference_samples_train=1,
              inference_samples_test=1,
              inference_samples_predict=1,
              beta=0.03)
    vae.compile(optimizer="adam", 
                loss={'y':tf.keras.losses.mean_squared_error,
                      'z_params':KLDivergence()},
                loss_weights={'y':1,
                              'z_params':vae.beta})
    vae.fit(x=mnist_labels_one_hot, y=mnist_digits, 
            callbacks=[#BetaScaling(method="linear", min_beta=0.1, max_beta=1),
                       savefig("figures")], 
            epochs=100, validation_split=0.1)

