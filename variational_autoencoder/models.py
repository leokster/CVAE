import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
if int(tf.__version__.replace(".", "")) < 240:
    from tensorflow.python.keras.engine.training import _minimize


def _get_input_shape(layer):
    return layer.layers[0].output_shape


def _get_output_len(layer):
    return len(layer.output)


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, prior, **kwargs):
        """
        Initialize the Conditional Variational Autoencoder model as subcalss of tf.keras.Model.
        In the following X is a tensor in the domain space and y is a tensor in the target space
        (including) batch sizes.

        Input parameters:
        :param encoder: tf.keras.Model taking a list of two tensors as input [X, y] and outputs
        a list of 3 tensors [Z_mean, Z_logvar, Z_smpl] in the latent space, where Z_mean is the mean
        of a Gaussian, Z_logvar the logvariance of the Gaussian and Z_smpl one sample of the corresponding
        distribution (can be realized with the Sampling layer).
        :pram prior: tf.keras.Model taking a tensor X as input and outputs a list of 3 tensors as the encoder
        does
        :param decoder: tf.keras.Model taking a list of two tensors [X, Z]
        where Z are samples of the latent space and maps it to the target
        space y. Note: if the decoder has multiple outpts (mor than one tensor) during perdiction only the last one
        is returned (if training=False). This allows to build parametric distributions. See example below.


        Example:

        def make_prior(latent_dim=2):
            prior_input = tf.keras.layers.Input(13)
            # add some layers
            # x = tf.keras.layers.Dense(64)
            z_mu = tf.keras.layers.Dense(latent_dim)(x)
            z_logvar = tf.keras.layers.Dense(latent_dim)(x)
            z_smpl = Sampling()([z_mu, z_logvar])
            return tf.keras.models.Model(prior_input, [z_mu, z_logvar, z_smpl])

        def make_encoder(latent_dim=2):
            x_input = tf.keras.layers.Input(13)
            y_input = tf.keras.layers.Input(1)
            x = tf.keras.layers.concatenate([x_input, y_input])
            # add some layers
            # x = tf.keras.layers.Dense(64)
            z_mu = tf.keras.layers.Dense(latent_dim)(x)
            z_logvar = tf.keras.layers.Dense(latent_dim)(x)
            z_smpl = Sampling()([z_mu, z_logvar])
            return tf.keras.models.Model([x_input, y_input], [z_mu, z_logvar, z_smpl])

        def make_decoder(latent_dim=2):
            x_input = tf.keras.layers.Input(13)
            z_input = tf.keras.layers.Input(latent_dim)
            x = tf.keras.layers.concatenate([x_input, z_input])
            # add some layers
            # x = tf.keras.layers.Dense(64)
            y_mu = tf.keras.layers.Dense(1)(x)
            y_logvar = tf.keras.layers.Dense(1)(x)
            y_smpl = Sampling()([y_mu, y_logvar])
            y_params = tf.keras.layers.concatenate([y_mu, y_logvar], axis=0)
            return tf.keras.models.Model([x_input, z_input], [y_params, y_smpl])
        """
        self.beta = tf.Variable(kwargs.pop("beta", 1) * 1.0, trainable=False)
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

        if _get_output_len(self.prior) != 3:
            raise ValueError("The prior must contain 4 output dimensions. It only",
                             "contains {} output dimensions".format(_get_output_len(self.prior)))

        if _get_output_len(self.encoder) != 3:
            raise ValueError("The encoder must contain 4 output dimensions. It only",
                             "contains {} output dimensions".format(_get_output_len(self.encoder)))

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

        if int(tf.__version__.replace(".","")) < 240:
            _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                      self.trainable_variables)
        else:
            self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """The logic for one evaluation step.
        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.make_test_function`.
        This function should contain the mathematical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_test_function`, which can also be overridden.
        Arguments:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned.
        """
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred_inference = self(x, training=False, verbose=True)
        #y_pred = self((x,y), training=True)

        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred_inference, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred_inference, sample_weight)
        return {m.name: m.result() for m in self.metrics}



    def call(self, data, training=False, sample_size=1, verbose=0):
        # if in training mode
        if training:
            # unpack data
            if isinstance(data, list) or isinstance(data, tuple):
                if len(data) != 2:
                    raise ValueError("data must be length 2 tuple or list of type (data_x, data_y)")
                else:
                    data_x, data_y = data
            else:
                raise ValueError("data must be length 2 tuple or list of type (data_x, data_y)")
                #data_x = data
                #data_y = None

            # run encoder on data_x and data_y
            z_mean, z_log_var, z = self.encoder([data_x, data_y])

            # run prior on data_x
            z_prior_mean, z_prior_log_var, zz = self.prior(data_x)

            # compute Kullback–Leibler divergence between prior and encoder
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - z_prior_log_var - tf.exp(-z_prior_log_var) * (
                    tf.exp(z_log_var) + tf.square(z_mean - z_prior_mean)))

            # run decoder on data_x and z where z is sampled from encoder
            reconstruction = self.decoder([data_x, z])

            # add loss for gradient computation
            self.add_loss(self.beta * kl_loss)

            # add metrices for training logs
            self.add_metric(kl_loss, aggregation='mean', name='kl_loss')
            self.add_metric(self.beta, aggregation='mean', name='beta')
            return reconstruction

        # if in inference mode
        else:
            _, _, z = self.prior(data)
            reconstruction = self.decoder([data, z])

            if isinstance(reconstruction, list):
                y_sampled = reconstruction[-1]
            else:
                y_sampled = reconstruction

            if verbose == 0:
                return y_sampled
            if verbose == 1:
                return reconstruction
