from tensorflow.keras.losses import Loss
import tensorflow as tf

class FullLikelihood(Loss):
    def call(self, y_true, y_pred):
        reconstruction_loss = tf.reduce_mean(
            0.5 * tf.square(y_true - y_pred[0]) * tf.exp(-y_pred[1]) +
            y_pred[1]
        )
        return reconstruction_loss