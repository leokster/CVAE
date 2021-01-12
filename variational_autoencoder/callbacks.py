import tensorflow as tf
import numpy as np

class BetaScaling(tf.keras.callbacks.Callback):
    def __init__(self, method="linear", min_beta=0.00001, max_beta=1, **kwargs):
        super(BetaScaling, self).__init__()

        if method == "linear":
            self.method = lambda epoch: epoch*(max_beta-min_beta)/self.params['epochs']+min_beta
        elif method == "exponential":
            raise ValueError("exponential is not yet implemented")
            #self.method = lambda epoch: (np.exp(-3*epoch/self.params['epochs'])+1)/2
        elif callable(method):
            self.method = method
        else:
            raise TypeError("No valid method passed. Method must be either the string 'linear', 'exponential' or",
                            "function which takes epoch number as input and returns",
                            "a weight for the reconstruction loss weight between 0 and 1")

    def on_epoch_begin(self, epoch, logs=None):
        self.model.beta.assign(tf.Variable(self.method(epoch)*1.0, trainable=False, dtype="float32"))
