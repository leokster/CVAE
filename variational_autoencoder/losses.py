from tensorflow.keras.losses import Loss
import tensorflow as tf

def _get_slice_last_dim(ten, idx):
    dims = len(ten.shape)
    return tf.slice(ten, [0]*(dims-1)+[idx], [-1]*(dims-1)+[1])

class FullLikelihood(Loss):
    def call(self, y_true, y_pred):
        mean = _get_slice_last_dim(y_pred,0)
        sd = _get_slice_last_dim(y_pred,1)
        reconstruction_loss = tf.reduce_mean(
            0.5 * tf.square(y_true - mean) * tf.exp(-sd) + sd
        )
        return reconstruction_loss