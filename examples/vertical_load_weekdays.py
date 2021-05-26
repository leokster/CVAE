#from total_energy_consumption_analysis.total_energy_2019 import scaler

import tensorflow as tf
from variational_autoencoder.models import VAE
from variational_autoencoder.layers import Sampling
from variational_autoencoder.layers import StackNTimes
from variational_autoencoder.evaluations import get_p_vals
from variational_autoencoder.callbacks import BetaScaling
from variational_autoencoder.losses import GaussianLikelihood, KLDivergence

def make_prior(latent_dim=2, input_dim=1):
    sample_input = tf.keras.Input(shape=[], batch_size=1)
    samples = tf.squeeze(sample_input)
    
    input_x = tf.keras.Input(input_dim)
    x = tf.keras.layers.Dense(128, activation="relu")(input_x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_mean = StackNTimes(axis=1)(z_mean, samples)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = StackNTimes(axis=1)(z_log_var, samples)
    z_params = tf.stack([z_mean, z_log_var], axis=-1)
    z = Sampling()([z_mean, tf.exp(z_log_var/2)])
    return tf.keras.models.Model([input_x, sample_input], [z_params, z])


def make_encoder(latent_dim=2, output_dim=1, input_dim=1):
    sample_input = tf.keras.Input(shape=[], batch_size=1)
    samples = tf.squeeze(sample_input)
    
    input_x = tf.keras.Input(input_dim)
    input_y = tf.keras.Input(output_dim)
    x = tf.keras.layers.Concatenate()([input_x, input_y])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_mean = StackNTimes(axis=1)(z_mean, samples)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = StackNTimes(axis=1)(z_log_var, samples)
    z_params = tf.stack([z_mean, z_log_var], axis=-1)
    z = Sampling()([z_mean, tf.exp(z_log_var/2)])
    return tf.keras.models.Model([input_x, input_y, sample_input], [z_params, z])


def make_decoder(latent_dim=2,  output_dim=1, input_dim=1):
    sample_input = tf.keras.Input(shape=[], batch_size=1)
    samples = tf.squeeze(sample_input)
    
    input_z = tf.keras.Input((None, latent_dim))
    input_x = tf.keras.Input(input_dim)
    x = StackNTimes(axis=1)(input_x, samples)
    x = tf.keras.layers.Concatenate()([x, input_z])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    y_mean = tf.keras.layers.Dense(output_dim, name="y_mean")(x)
    y_log_var = tf.keras.layers.Dense(output_dim, name="y_log_var")(x)
    y_params = tf.stack([y_mean, y_log_var], axis=-1)
    y = Sampling()([y_mean, tf.exp(y_log_var/2)])
    return tf.keras.models.Model([input_x, input_z, sample_input], [y_params, y])



def autoregressive_prediction(x, horizon, model):
    for i in range(horizon):
        x = np.concatenate([x, model.predict(x[:,-24:])], axis=1)
    return x[:,-horizon:]



if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    import holidays

    data = pd.read_csv("../data/swissgrid_total_load.csv", index_col=0)
    ch_holidays = holidays.CountryHoliday("CH")

    scaler = StandardScaler()
    data["MW"] = scaler.fit_transform(np.array(data["MW"]).reshape(-1,1)).flatten()
    data["time"] = pd.to_datetime(pd.Series(data.index, index=data.index)).dt.time
    data["date"] = pd.to_datetime(pd.to_datetime(pd.Series(data.index, index=data.index)).dt.date)
    data_y = data.reset_index(drop=True)[~pd.Series(data.index).duplicated()].pivot(columns="time", index="date", values="MW").dropna()

    weekday = tf.one_hot( pd.to_datetime(pd.Series(data_y.index)).dt.weekday, depth=7).numpy()
    season = pd.to_datetime(pd.Series(data_y.index)).dt.dayofyear.to_numpy().reshape(-1,1)/365
    holiday_indicator = pd.to_datetime(pd.Series(data_y.index)).apply(lambda x: x in ch_holidays).to_numpy().reshape(-1,1)

    data_x = np.concatenate([weekday, season, holiday_indicator], axis=1)


    for i in range(10):
        latent_dim = (i+1)*3
        prior = make_prior(latent_dim=latent_dim, input_dim=data_x.shape[1])
        encoder = make_encoder(latent_dim=latent_dim, output_dim=data_y.shape[1], input_dim=data_x.shape[1])
        decoder = make_decoder(latent_dim=latent_dim, output_dim=data_y.shape[1], input_dim=data_x.shape[1])

        model = VAE(encoder, decoder, prior)
        model.compile(optimizer="adam",
                      loss={'y_params':GaussianLikelihood(sample_axis=True),
                            'z_params':KLDivergence()},
                      loss_weights={'y_params':1,
                                    'z_params':model.beta})

        model.fit(data_x, data_y, epochs=50, callbacks=BetaScaling(method="linear"))

        today = np.array([0,0,0,0,1,0,0,0.94520548,0])
        test_x = today.reshape(1,-1)
        model_output = model(test_x, samples=10000, verbose=True)
        res = scaler.inverse_transform(
            model_output['y_params'][0,...,0]
        )
        res2 = scaler.inverse_transform(
            model_output['y'][0]
        )

        fig, ax = plt.subplots(1, 2, figsize=(30, 10))
        pd.DataFrame(res.transpose()+500, index=data_y.columns).iloc[:, :30].plot(ax=ax[0])
        pd.DataFrame(res2.transpose()+500, index=data_y.columns).iloc[:, :30].plot(ax=ax[1])
        ax[0].set(ylabel="Power [MW]", xlabel="Time", ylim=(3000,8000))
        ax[0].set_title("Load Profile Samples ({}d Latent Space)".format(latent_dim), fontsize=20)
        ax[1].set(ylabel="Power [MW]", xlabel="Time", ylim=(3000,8000))
        ax[1].set_title("Load Profile Samples ({}d Latent Space) with Noise".format(latent_dim), fontsize=20)
        ax[0].legend().remove()
        ax[1].legend().remove()
        plt.show()
        fig.savefig("latent_{}.eps".format(latent_dim))

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    test_y = scaler.inverse_transform(data_y[np.logical_and(np.logical_and(data_x[:,4]==1, data_x[:,7]>0.92), data_x[:,-1]==0)])
    ax.plot(test_y.transpose())
    ax.set(ylabel="Power [MW]", xlabel="Time", ylim=(3000,8000))
    ax.set_title("True Load Profiles", fontsize=20)
    ax.legend().remove()
    fig.savefig("true.eps")

    scatter_data = pd.DataFrame(res.transpose(), index=data_y.columns).unstack().reset_index().rename(columns={"level_1": "time", 0: "value"}).drop(
        columns=["level_0"])
    fig, ax = plt.subplots(1,1, figsize=(20,10))
    #sns.scatterplot(data=scatter_data,
    #                x="time",
    #                y="value", alpha=0.01, color="Gray", ax=ax)

    sns.violinplot(data=scatter_data,
                    x="time",
                    y="value", ax=ax, color="Yellow")
    #sns.lineplot(x=data_y.columns, y=res.mean(axis=0), ax=ax)
    #ax.set(y_limit=(2000,7000))
    sns.lineplot(data=res[:500,:].transpose())
    sns.lineplot(data=data_y[(data_x[:,0]==1)].reset_index(drop=True).to_numpy().transpose())
    plt.show()

    tmp = model.predict(data_x.repeat(1000,0)).reshape(-1,1000,24)
    p_comp_smpl = tf.Variable(tmp.transpose(0,2,1).reshape(-1,1000), dtype="float64")
    p_comp_true = tf.Variable(data_y.to_numpy().flatten(), dtype="float64")
    p_vals = get_p_vals(p_comp_smpl, p_comp_true)
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    sns.lineplot(x=p_vals.numpy().flatten(), y=np.linspace(0,1,len(p_vals)), ax=ax)
    plt.show()




    fig, ax = plt.subplots(1,1)
    ax.plot(data_x[9:10,-10:].flatten())

    df_hist = pd.DataFrame(model.predict(data_x[0:1,:].repeat(1000,0)).transpose()).stack().reset_index().drop(columns=["level_1"]).rename(columns={"level_0":"offset",0:"val"})
    import seaborn as sns
    sns.displot(df_hist, x="val", hue="offset", kind="kde")
    plt.show()


