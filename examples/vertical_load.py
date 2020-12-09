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


def make_encoder(latent_dim=2, output_dim=1):
    input_x = tf.keras.Input(24)
    input_y = tf.keras.Input(output_dim)
    x = tf.keras.layers.Concatenate(axis=1)([input_x, input_y])
    x = tf.keras.layers.Reshape((24+output_dim,1))(x)
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


def make_decoder(latent_dim=2,  output_dim=1):
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
    y_mean = tf.keras.layers.Dense(output_dim, name="y_mean")(x)
    y_log_var = tf.keras.layers.Dense(output_dim, name="y_log_var")(x)
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

    history_len = 24
    prediction_len = 10

    data_window = pd.DataFrame({i: data["MW"].shift(-i) for i in range(history_len+prediction_len)}).dropna()

    latent_dim = 30
    prior = make_prior(latent_dim)
    encoder = make_encoder(latent_dim, output_dim=prediction_len)
    decoder = make_decoder(latent_dim, output_dim=prediction_len)

    x_train = np.array(data_window.iloc[:,:history_len]).reshape(-1,history_len)
    y_train = np.array(data_window.iloc[:,history_len:]).reshape(-1,prediction_len)

    encoder([tf.Variable(x_train), tf.Variable(y_train)])
    prior(x_train)

    model = VAE(encoder, decoder, prior)
    model.compile(optimizer="adam")

    model.fit(x_train, y_train, epochs=100, callbacks=ReduceReconstructionWeight())


    res =  scaler.inverse_transform(
        autoregressive_prediction(x_train[10,:].reshape(1,-1).repeat(1, axis=0), 50, model))
    pd.DataFrame(res.transpose()).plot()
    plt.show()

fig, ax = plt.subplots(1,1)
ax.plot(x_train[9:10,-10:].flatten())

df_hist = pd.DataFrame(model.predict(x_train[0:1,:].repeat(1000,0)).transpose()).stack().reset_index().drop(columns=["level_1"]).rename(columns={"level_0":"offset",0:"val"})
import seaborn as sns
sns.displot(df_hist, x="val", hue="offset", kind="kde")
plt.show()