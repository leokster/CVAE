import tensorflow as tf
from variational_autoencoder.layers import Sampling
from variational_autoencoder.models import VAE
from variational_autoencoder.callbacks import BetaScaling
from variational_autoencoder.losses import FullLikelihood
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def make_prior(latent_dim=2):
    prior_input = tf.keras.layers.Input(13)
    x = tf.keras.layers.Dense(512)(prior_input)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    z_mu = tf.keras.layers.Dense(latent_dim)(x)
    z_logvar = tf.keras.layers.Dense(latent_dim)(x)
    z_smpl = Sampling()([z_mu, z_logvar])
    return tf.keras.models.Model(prior_input, [z_mu, z_logvar, z_smpl])


def make_encoder(latent_dim=2):
    x_input = tf.keras.layers.Input(13)
    y_input = tf.keras.layers.Input(1)
    x = tf.keras.layers.concatenate([x_input, y_input])

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    z_mu = tf.keras.layers.Dense(latent_dim)(x)
    z_logvar = tf.keras.layers.Dense(latent_dim)(x)
    z_smpl = Sampling()([z_mu, z_logvar])
    return tf.keras.models.Model([x_input, y_input], [z_mu, z_logvar, z_smpl])



def make_decoder(latent_dim=2):
    x_input = tf.keras.layers.Input(13)
    z_input = tf.keras.layers.Input(latent_dim)
    x = tf.keras.layers.concatenate([x_input, z_input])
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    y_mu = tf.keras.layers.Dense(1)(x)
    y_logvar = tf.keras.layers.Dense(1)(x)
    y_smpl = Sampling()([y_mu, y_logvar])
    return tf.keras.models.Model([x_input, z_input], [y_mu, y_logvar, y_smpl])

def read_index(path):
    lines = open(path).readlines()
    return list(map(lambda x: int(x.replace("\n","")), lines))

if __name__ == "__main__":
    import numpy as np
   (x_data, y_data),_ = tf.keras.datasets.boston_housing.load_data(
        path="boston_housing.npz", test_split=0, seed=0
    )
   y_data = y_data.reshape(-1,1)



   res = []
   for i in range(20):
       test_index = read_index( "mc_dropout/data/index_test_{}.txt".format(i))
       train_index = read_index("mc_dropout/data/index_train_{}.txt".format(i))

       x_train = x_data[train_index]
       x_test = x_data[test_index]
       y_train = y_data[train_index]
       y_test = y_data[test_index]

       x_std = np.std(x_train, 0)
       x_mean = np.mean(x_train,0)

       y_std = np.std(y_train, 0)
       y_mean = np.mean(y_train,0)

       x_train = (x_train-x_mean)/x_std
       y_train = (y_train-y_mean)/y_std
       x_test = (x_test-x_mean)/x_std
       #y_test = (y_test-y_mean)/y_std


       latent_dim = 10


       prior = make_prior(latent_dim)
       encoder = make_encoder(latent_dim)
       decoder = make_decoder(latent_dim)

       vae = VAE(encoder=encoder, decoder=decoder, prior=prior, beta=0)
       vae.compile(loss=FullLikelihood(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

       vae.fit(x_train, y_train, epochs=200, callbacks = [BetaScaling(min_beta=0.0001,max_beta=1)])

       #standard_pred = ((vae.predict(x_test.repeat(100, axis=0))*y_std)+y_mean).reshape(-1,100)
       standard_pred = (((vae.predict(x_test))*y_std)+y_mean)

       res.append(
            np.mean(
                (standard_pred-y_test)**2
                   )**0.5
       )
       print(res)