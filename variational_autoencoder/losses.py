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
        if type(self.normalize) == int or type(self.normalize) == float:
            loss /= self.normalize
        return loss
    
class KLDivergence(Loss):
    def __init__(self, normalize=True, **kwargs):
        super(KLDivergence, self).__init__(**kwargs)
        self.normalize = normalize
    
    def call(self, y_true, z_params):
        mu_pri = z_params[...,0]
        lv_pri = z_params[...,1]
        mu_enc = z_params[...,2]
        lv_enc = z_params[...,3]
        loss = ((tf.square(mu_enc - mu_pri) + tf.exp(lv_enc))/tf.exp(lv_pri)
                + lv_pri - lv_enc - 1)/2
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        if self.normalize==True:
            loss /= y_true.shape[-1]
        if type(self.normalize) == int or type(self.normalize) == float:
            loss /= self.normalize
        return loss