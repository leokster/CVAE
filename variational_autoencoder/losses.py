from tensorflow.keras.losses import Loss
import tensorflow as tf
import math
from .layers import StackNTimes

class GaussianLikelihood(Loss):
    """Gaussian likelihood loss with optional normalization. Sums over
    feature axis. By default, divides by the last feature dimension 
    inferred from y_true, or divides by a passed scalar. 
    
    Takes y_pred as a tensor with mean and log-variance stacked 
    in the last dimension.
    
    Optionally stacks the y_true tensor in dimension 1 for compatibility
    with a sampling axis in y_pred.
    
    Inputs to constructor:
    normalize   - A boolean, integer or float. Defaults to true. If true,
                  divides the loss by the feature dimensio inferred from
                  y_true. If false, does not normalize. If integer or float,
                  divides loss by this value.
    sample_axis - Option to stack y_true in dimension 1 for compatibility
                  with a sampling axis in y_pred. Defaults to false."""
                  
    def __init__(self, normalize=True, sample_axis=False, **kwargs):
        super(GaussianLikelihood, self).__init__(**kwargs)
        self.normalize = normalize
        self.sample_axis = sample_axis
        if self.sample_axis:
            self.stack_inputs = StackNTimes(axis=1)
    
    def call(self, y_true, y_pred):
        """Inputs:
        y_true - Tensor with real data
        y_pred - Tensor with mean and log-variance stacked in last dimension.
                 If sample_axis is false, shape must be the shape of y_true
                 with an extra dimension of 2 at the end for mean and
                 log-variance. If sample_axis is true, must be this shape
                 but with an extra sampling axis in dimension 1."""
                 
        mean = y_pred[..., 0]
        log_var = y_pred[..., 1]
        
        if self.sample_axis:
            y_true = self.stack_inputs(y_true, mean.shape[1])
        
        loss =  (tf.square(y_true - mean)/tf.exp(log_var) 
                 + log_var + math.log(2*math.pi))/2
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        if self.normalize==True:
            loss /= y_true.shape[-1]
        if type(self.normalize) == int or type(self.normalize) == float:
            loss /= self.normalize
        return loss
    
class KLDivergence(Loss):
    """Kullback-Leibler Divergence regularization with optional normalization. 
    Sums over feature axis. By default, divides by the last feature dimension 
    inferred from y_true, or divides by a passed scalar. 
    
    Takes z_params as a tensor with mean and log-variance of two distributions
    stacked in the last dimension, for a total last dimensino of 4.
    
    Inputs to constructor:
    normalize   - A boolean, integer or float. Defaults to true. If true,
                  divides the loss by the feature dimensio inferred from
                  y_true. If false, does not normalize. If integer or float,
                  divides loss by this value."""

    def __init__(self, normalize=True, **kwargs):
        super(KLDivergence, self).__init__(**kwargs)
        self.normalize = normalize
    
    def call(self, y_true, z_params):
        """Inputs:
        y_true   - Tensor with real data, for compatibility with keras loss
                   API and to infer default normalization.
        z_params - Tensor with mean and log-variance of two distributions
                   stacked in last dimension, in the following order:
                   [mean1, log_var1, mean2, log_var2]."""
        mu_pri = z_params[..., 0]
        lv_pri = z_params[..., 1]
        mu_enc = z_params[..., 2]
        lv_enc = z_params[..., 3]
        loss = ((tf.square(mu_enc - mu_pri)
                 + tf.exp(lv_enc))/tf.exp(lv_pri)
                + lv_pri - lv_enc - 1)/2
        
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        
        if self.normalize==True:
            loss /= y_true.shape[-1]
        if type(self.normalize) == int or type(self.normalize) == float:
            loss /= self.normalize
        return loss