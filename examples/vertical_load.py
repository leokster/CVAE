import sys

import tensorflow as tf
from variational_autoencoder.models import VAE
from variational_autoencoder.layers import Sampling
from variational_autoencoder.callbacks import BetaScaling

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




if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns

    #load data
    data = pd.read_csv("data/swissgrid_total_load.csv", index_col=0)

    #initialize the scaler for the load data
    scaler = StandardScaler()
    data["MW"] = scaler.fit_transform(np.array(data["MW"]).reshape(-1,1)).flatten()

    #number of past datapoints which should be considered for prediction
    history_len = 24

    #number of future datapoints which should be predicted
    prediction_len = 24

    #dimension of latent space (Z space)
    latent_dim = 30

    #initialize the three networks
    prior = make_prior(latent_dim)
    encoder = make_encoder(latent_dim, output_dim=prediction_len)
    decoder = make_decoder(latent_dim, output_dim=prediction_len)

    #window size (based on history_len and prediction_en)
    data_window = pd.DataFrame({i: data["MW"].shift(-i) for i in range(history_len+prediction_len)}).dropna()

    #split dataframe into x and y
    x = np.array(data_window.iloc[:,:history_len]).reshape(-1,history_len)
    y = np.array(data_window.iloc[:,history_len:]).reshape(-1,prediction_len)

    #build VAE model
    model = VAE(encoder, decoder, prior)
    model.compile(optimizer="adam")

    #fit model
    model.fit(x, y, epochs=100, callbacks=BetaScaling())

    #choose random datapoint
    start_point = 3551

    #evaluet datapoint
    res = scaler.inverse_transform(
        model(x[start_point,:].reshape(1,-1).repeat(100, axis=0), verbose=1)[2]
    )

    #make plot of samples
    pd.DataFrame(res[0:10,:]).transpose().plot()
    plt.show()

    #make scatterplot of VAE output + ground truth
    scatter_data = pd.DataFrame(res.transpose()).unstack().reset_index().rename(columns={"level_1": "time", 0: "value"}).drop(
        columns=["level_0"])
    scatter_data["time"] += history_len
    fig, ax = plt.subplots(1,1)
    sns.scatterplot(data=scatter_data,
                    x="time",
                    y="value", alpha=0.1, color="Gray", ax=ax)
    true_vals = scaler.inverse_transform(np.concatenate([x[start_point,:],
                                                         y[start_point:start_point + prediction_len,0].flatten()]))
    sns.lineplot(x=list(range(0, history_len+prediction_len)), y=true_vals, ax=ax)
    plt.show()
    fig.savefig("powergrid.png")

