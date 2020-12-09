import tensorflow as tf
import numpy as np

nbins = 100

def __get_p_vals(data_array):
    y_pred = data_array[:-1]
    y_true = data_array[-1]

    support = [tf.reduce_min(data_array), tf.reduce_max(data_array)]
    bins = tf.histogram_fixed_width_bins( y_pred, support, nbins=nbins, dtype=tf.dtypes.int32, name=None)
    probs = tf.math.bincount(bins, minlength=nbins, maxlength=nbins)/y_pred.shape[0]
    bin_y_true = tf.histogram_fixed_width_bins(y_true, support, nbins=nbins, dtype=tf.dtypes.int32, name=None)
    res = tf.cast(tf.reduce_sum(probs[probs <= probs[bin_y_true]]), tf.float32)
    return res

@tf.function
def get_p_vals(pred, true):
    '''
    :param pred: predicted values in a 2d-tensor of dimensions (datapoints, samples)
    :param true: true values in a 1d-tensor of dimensions (datapoints)
    :return: p-values in a 1d-tensor of dimensions (timestamp)
    '''
    data_array = tf.concat([pred, tf.reshape(true, (-1,1))], axis=1)
    return tf.reshape(tf.sort(tf.map_fn(__get_p_vals, data_array, dtype=tf.float32)),(-1,))

@tf.function
def calc_area(p_vals):
    p_vals = tf.reshape(tf.sort(p_vals),(-1,))
    return tf.reduce_sum(tf.abs(p_vals-tf.linspace(float(0),float(1),len(p_vals))))/len(p_vals)


def calc_area_separated(p_vals):
    p_vals = tf.reshape(tf.sort(p_vals),(-1,))
    area_neg =  tf.reduce_sum(tf.minimum(p_vals-tf.linspace(float(0),float(1),len(p_vals)),0))/len(p_vals)
    area_pos =  tf.reduce_sum(tf.maximum(p_vals-tf.linspace(float(0),float(1),len(p_vals)),0))/len(p_vals)
    return area_pos, -area_neg

if __name__ == "__main__":

    smpls = np.random.normal(0.7,1, [1000, 1000])
    true_vals = np.random.lognormal(0, 1, 1000)

    p_vals = get_p_vals(smpls, true_vals)
    p_area = calc_area(p_vals)

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1, figsize=(10,4))
    lnsp = np.linspace(0,1,len(p_vals))
    ttmp = p_vals.numpy().flatten()
    sns.lineplot(data=pd.DataFrame({"lin":lnsp, "value":ttmp}), x="value", y="lin", ax=ax)
    sns.lineplot(x=[0,1], y=[0,1], ax=ax)
    ax.fill_between(ttmp, ttmp, lnsp, alpha=0.1)
    ax.set(ylim=(0, 1))
    ax.set(xlim=(0, 1))
    fig.savefig("p_vals.png")