import tensorflow as tf
import numpy as np

class ReduceReconstructionLoss(tf.keras.callbacks.Callback):
    def __init__(self, method="linear", **kwargs):
        super(ReduceReconstructionLoss, self).__init__()

        if method == "linear":
            self.method = lambda epoch: 1-0.5*epoch/self.params['epochs']
        elif method == "exponential":
            self.method = lambda epoch: (np.exp(-3*epoch/self.params['epochs'])+1)/2
        elif callable(method):
            self.method = method
        else:
            raise TypeError("no valid method. Method must be either the string 'linear' or",
                            "function which takes epoch number as input and returns",
                            "a weight for the reconstruction loss between 0 and 1")

    def on_epoch_begin(self, epoch, logs=None):
        self.model.reconstruction_weight.assign(tf.Variable(self.method(epoch)*1.0, trainable=False, dtype="float32"))
