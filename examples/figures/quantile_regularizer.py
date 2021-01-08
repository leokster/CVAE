import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.utils import losses_utils


class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self,
                 quantile_threshold=0.05,
                 regulairzer_weight=0.5,
                 reduction=losses_utils.ReductionV2.AUTO,
                 default_loss=tf.keras.losses.MeanAbsoluteError()
                 ):
        '''
        The QuantileLoss can be used on a neural network with at least 3 dimensional output
        tensor (batch_size, sample_size, feature_size).
        :param quantile_threshold: number between 0 and 1, denoting the quantile, from
        where on the additional regularization term should be added
        :param regulairzer_weight: The convex weight of the regularizaation term. The
        default_loss is weighted with (1-regularizer_weight).
        :param reduction: The reduction method.
        :param default_loss: The default loss e.g. L2 loss.
        '''
        super(QuantileLoss, self).__init__(reduction=reduction)
        self.regulairzer_weight = tf.Variable(regulairzer_weight, dtype="float32")
        self.quantile_threshold = tf.Variable(quantile_threshold, dtype="float32")
        self.default_loss = default_loss
        self.ncdf = tfp.bijectors.NormalCDF()
        self.final_loss = tf.Variable(0, dtype="float32")

    def add_default_loss(self, y_true, y_pred):
        # broadcasted version of the y_true such that it has the same shape
        # as the y_pred
        tmp = self.default_loss(tf.expand_dims(y_true, axis=1)
                                * tf.ones((1, *y_pred.shape[1:])), y_pred)
        self.final_loss.assign_add(tmp)
        return tmp

    def add_regularization(self, z_pval):
        tmp = -tf.reduce_mean(z_pval, axis=(-1, -2))
        self.final_loss.assign_add(tmp)
        return tmp

    def reset_loss(self):
        self.final_loss.assign(0)

    def call(self, y_true, y_pred, sample_weight=None):
        '''
        :param y_true: 2d tensor (batch_size, feature_size)
        :param y_pred: 3d tensor (batch_size, sample_size, feature_size)
        :return:
        '''
        #compute the mean on the prediction along sample axis
        mean = tf.reduce_mean(y_pred, 1)

        #compute standard deviation on the prediction along sample axis
        std = tf.math.reduce_std(y_pred, 1)

        z = (tf.squeeze(y_pred, -1) - mean) / std
        z_cdf = self.ncdf(z)
        z_pval = tf.minimum(z_cdf, 1 - z_cdf)

        in_range = tf.cast(tf.math.greater_equal(z_pval, self.quantile_threshold / 2), "float32")

        self.reset_loss()

        return tf.cond(tf.reduce_mean(in_range) <= 1 - self.quantile_threshold,
                       lambda: ((1 - self.regulairzer_weight) * self.add_default_loss(y_true,y_pred) +
                                self.regulairzer_weight * self.add_regularization(z_pval)),
                       lambda: self.add_default_loss(y_true, y_pred)
                       )

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

x_train = np.random.normal(0, 1, (1000, 16))
y_train = np.random.normal(0, 1, (1000, 1))

smpl_size = 100
x_input = tf.keras.layers.Input(16)
x = tf.expand_dims(x_input, axis=1)
x = x * tf.ones((1, smpl_size, 1))
x = tf.keras.layers.Dropout(0.2)(x, training=True)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x, training=True)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x, training=True)
x = tf.keras.layers.Dense(1)(x)
y_output = x
model = tf.keras.models.Model(x_input, y_output)
model.summary()
#tf.config.experimental_run_functions_eagerly(True)

model.compile(loss=QuantileLoss(quantile_threshold=0, regulairzer_weight=0.8), optimizer="adam")

model.fit(x_train, y_train, epochs=12, batch_size=16)

sns.distplot(model.predict(x_train)[:, 0, 0])
plt.show()


