from tensorflow.keras.losses import Loss
import tensorflow as tf
import math

class GaussianLikelihood(Loss):
    def __init__(self, normalize=True, **kwargs):
        super(GaussianLikelihood, self).__init__(**kwargs)
        self.normalize = normalize
    
    def call(self, y_true, y_pred):
        mean = y_pred[...,0]
        log_var = y_pred[...,1]
        loss =  (tf.square(y_true - mean)/tf.exp(log_var) 
                 + log_var + math.log(2*math.pi))/2
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        if self.normalize==True:
            loss /= mean.shape[-1]
        if isinstance(self.normalize, (int, float)):
            loss /= self.normalize
        return loss