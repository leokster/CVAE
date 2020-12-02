import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine.training import _minimize


def _get_input_shape(layer):
    return layer.layers[0].output_shape


def _get_output_len(layer):
    return len(layer.output)


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, prior, **kwargs):
        self.reconstruction_weight = tf.Variable(min(kwargs.pop("reconstruction_weight", 0.5), 1) * 1.0,
                                                 trainable=False)
        self.reconstruction_loss = kwargs.pop("reconstruction_loss", "likelihood")
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

        if _get_output_len(self.prior) != 3:
            raise ValueError("The prior must contain 4 output dimensions. It only",
                             "contains {} output dimensions".format(_get_output_len(self.prior)))

        if self.reconstruction_loss == "likelihood" and _get_output_len(self.decoder) != 3:
            raise ValueError("If likelihood is choosen as reconstruction loss, the decoder must",
                             "return three outputs [mean, logvar, sample]. Choose another reconstruction loss",
                             "or build different decoder")

        if self.reconstruction_loss != "likelihood" and not callable(self.reconstruction_loss):
            raise TypeError("reconstruction_loss must either be callable loss function or 'likelihood'")

    def train_step(self, data):
        """The logic for one training step.
        This method can be overridden to support custom training logic.
        This method is called by `Model.make_train_function`.
        This method should contain the mathemetical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.
        Arguments:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided. These utilities will be exposed
        # publicly.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self((x, y), training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        # For custom training steps, users can just write:
        #   trainable_variables = self.trainable_variables
        #   gradients = tape.gradient(loss, trainable_variables)
        #   self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        # The _minimize call does a few extra steps unnecessary in most cases,
        # such as loss scaling and gradient clipping.
        _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                  self.trainable_variables)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def call(self, data, training=False, sample_size=1):
        # if in training mode
        if training:
            # unpack data
            if isinstance(data, list) or isinstance(data, tuple):
                if len(data) != 2:
                    raise ValueError("data must be length 2 tuple or list of type (data_x, data_y)")
                else:
                    data_x, data_y = data
            else:
                data_x = data
                data_y = None

            # run encoder on data_x and data_y
            z_mean, z_log_var, z = self.encoder([data_x, data_y])

            # run prior on data_x
            z_prior_mean, z_prior_log_var, zz = self.prior(data_x)

            # compute Kullback–Leibler divergence between prior and encoder
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - z_prior_log_var - tf.exp(-z_prior_log_var) * (
                    tf.exp(z_log_var) + tf.square(z_mean - z_prior_mean)))

            # run decoder on data_x and z where z is sampled from encoder
            reconstruction = self.decoder([data_x, z])

            # compute likelihood of the true data_y value under the decoder distribution
            if self.reconstruction_loss == "likelihood":
                reconstruction_loss = tf.reduce_mean(
                    0.5 * tf.square(data_y - reconstruction[0]) * tf.exp(-reconstruction[1]) +
                    reconstruction[1]
                )
            else:
                reconstruction_loss = self.reconstruction_loss(data_y, reconstruction)

            # compute weighted sum of the two losses
            total_loss = self.reconstruction_weight * reconstruction_loss + \
                         (1 - self.reconstruction_weight) * kl_loss

            # add loss for gradient computation
            self.add_loss(total_loss)

            # add metrices for training logs
            self.add_metric(reconstruction_loss, aggregation='mean', name='r_loss')
            self.add_metric(kl_loss, aggregation='mean', name='kl_loss')
            self.add_metric(self.reconstruction_weight, aggregation='mean', name='rec_weight')
            return 0

        # if in inference mode
        else:
            _, _, z = self.prior(data)
            reconstruction = self.decoder([data, z])

            if isinstance(reconstruction, list):
                y_sampled = reconstruction[-1]
            else:
                y_sampled = reconstruction

            return y_sampled
