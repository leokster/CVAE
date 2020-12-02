import sys
sys.path.append("/Users/tim/OneDrive - Swissgrid AG/projects/variational_autoencoder")

import tensorflow as tf
from variational_autoencoder.models import VAE
from variational_autoencoder.layers import Sampling
from variational_autoencoder.callbacks import ReduceReconstructionWeight

def make_prior(latent_dim=2):
    input_x = tf.keras.Input(24)
    x = tf.keras.layers.Reshape((24,1))(input_x)
    x = tf.keras.layers.Conv1D(10,4)(x)
    x = tf.keras.layers.Conv1D(10,4)(x)
    x = tf.keras.layers.Conv1D(10,4)(x)
    x = tf.keras.layers.MaxPool1D(4)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.models.Model(input_x, [z_mean, z_log_var, z])


def make_encoder(latent_dim=2):
    input_x = tf.keras.Input(24)
    input_y = tf.keras.Input(1)
    x = tf.keras.layers.Concatenate(axis=1)([input_x, input_y])
    x = tf.keras.layers.Reshape((25,1))(x)
    x = tf.keras.layers.Conv1D(10,4)(x)
    x = tf.keras.layers.Conv1D(10,4)(x)
    x = tf.keras.layers.Conv1D(10,4)(x)
    x = tf.keras.layers.MaxPool1D(4)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.models.Model([input_x, input_y], [z_mean, z_log_var, z])


def make_decoder(latent_dim=2):
    input_z = tf.keras.Input(latent_dim)
    input_x = tf.keras.Input(24)
    x = tf.keras.layers.Reshape((24,1))(input_x)
    x = tf.keras.layers.Conv1D(10,4)(x)
    x = tf.keras.layers.Conv1D(10,4)(x)
    x = tf.keras.layers.Conv1D(10,4)(x)
    x = tf.keras.layers.MaxPool1D(4)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Concatenate(axis=1)([x,input_z])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    y_mean = tf.keras.layers.Dense(1, name="y_mean")(x)
    y_log_var = tf.keras.layers.Dense(1, name="y_log_var")(x)
    y = Sampling()([y_mean, y_log_var])
    return tf.keras.models.Model([input_x, input_z], [y_mean, y_log_var, y])



def autoregressive_prediction(x, horizon, model):
    for i in range(horizon):
        x = np.concatenate([x, model.predict(x[:,-24:])], axis=1)
    return x



if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    data = pd.read_csv("data/swissgrid_total_load.csv", index_col=0)

    scaler = StandardScaler()
    data["MW"] = scaler.fit_transform(np.array(data["MW"]).reshape(-1,1)).flatten()

    data_window = pd.DataFrame({i: data["MW"].shift(24-i) for i in range(25)}).dropna()

    latent_dim = 2
    prior = make_prior(2)
    encoder = make_encoder(2)
    decoder = make_decoder(2)

    x_train = np.array(data_window.iloc[:,:24]).reshape(-1,24)
    y_train = np.array(data_window.iloc[:,24]).reshape(-1,1)

    encoder([tf.Variable(x_train), tf.Variable(y_train)])
    prior(x_train)

    model = VAE(encoder, decoder, prior)
    model.compile(optimizer="adam")

    model.fit(x_train, y_train, epochs=25, callbacks=ReduceReconstructionWeight())


    res =  scaler.inverse_transform(autoregressive_prediction(x_train[0,:].reshape(1,-1).repeat(10, axis=0), 20, model))

    pd.DataFrame(res.transpose()).plot()

    plt.show()