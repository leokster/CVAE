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
    def __init__(self, encoder, decoder, prior, 
                 kl_loss=None, r_loss=None, latent_dim=None, 
                 training_mode_samples=1, 
                 inference_samples_train=10,
                 inference_samples_test=10, 
                 inference_samples_predict=1000,
                 **kwargs):
        """
        Initialize the Conditional Variational Autoencoder model as subclass of tf.keras.Model.
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
        space y. Note: if the decoder has multiple outputs (more than one tensor) during perdiction only the last one
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
        self.kl_loss = kl_loss
        self.r_loss = r_loss
        self.latent_dim = latent_dim
        self.training_mode_samples = training_mode_samples
        self.inference_samples_train = inference_samples_train
        self.inference_samples_test = inference_samples_test
        self.inference_samples_predict = inference_samples_predict

        #if _get_output_len(self.prior) != 3:
        #    raise ValueError("The prior must contain 3 output dimensions. It only",
        #                     "contains {} output dimensions".format(_get_output_len(self.prior)))

        #if _get_output_len(self.encoder) != 3:
        #    raise ValueError("The encoder must contain 3 output dimensions. It only",
        #                     "contains {} output dimensions".format(_get_output_len(self.encoder)))

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
            y_pred = self((x, y), training=True, 
                          samples=self.training_mode_samples)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)

        if int(tf.__version__.replace(".","")) < 240:
            _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                      self.trainable_variables)
        else:
            self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        
        # Run in inference mode for other metrics
        if self.compiled_metrics._metrics is not None:
            y_pred_inference = self(x, training=False, 
                                    samples=self.inference_samples_train,
                                    verbose=True)
            self.compiled_metrics.update_state(y, y_pred_inference, 
                                               sample_weight)

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

        # Run in training mode to calculate loss
        y_pred = self((x,y), training=True)
        
        # Updates stateful loss metrics.
        self.compiled_loss(y, y_pred, sample_weight, 
                           regularization_losses=self.losses)
        
        # Run in inference mode for other metrics
        if self.compiled_metrics._metrics is not None:
            y_pred_inference = self(x, training=False, 
                                    samples=self.inference_samples_test, 
                                    verbose=True)
            self.compiled_metrics.update_state(y, y_pred_inference, 
                                               sample_weight)
        
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        """The logic for one inference step.
        This method can be overridden to support custom inference logic.
        his method is called by `Model.make_predict_function`.
        his method should contain the mathematical logic for one step of inference.
        This typically includes the forward pass.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_predict_function`, which can also be overridden.
        Args:
            data: A nested structure of `Tensor`s.
        Returns:
            The result of one inference step, typically the output of calling the
            `Model` on data.
    """
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        
        return self(x, training=False, samples=self.inference_samples_predict)

    def build(self, input_shape):
        # Instantiate networks if passed as subclasses
        if isinstance(self.encoder, type):
            self.encoder = self.encoder(latent_dim=self.latent_dim)
        if isinstance(self.prior, type):
            self.prior = self.prior(latent_dim=self.latent_dim)
        if isinstance(self.decoder, type):
            assert isinstance(input_shape, (list, tuple)), (
                '''Cannot infer decoder output shape from x data only.
                Please call model on (x,y) in training mode to build'''
                )
            self.output_dim = input_shape[-1][-1]
            self.decoder = self.decoder(output_dim=self.output_dim)
            
        # Instantiate losses if passed as subclasses
        if isinstance(self.kl_loss, type):
            self.kl_loss = self.kl_loss(normalize=self.output_dim)
        if isinstance(self.r_loss, type):
            self.r_loss = self.r_loss(normalize=self.output_dim)

    def call(self, data, training=False, samples=1, verbose=False):
        
        # Training mode
        if training:
            
            # Unpack data
            if isinstance(data, (list, tuple)):
                if len(data) != 2:
                    raise ValueError('''Data must be length 2 tuple or list of 
                                     type (data_x, data_y)''')
                else:
                    data_x, data_y = data
            else:
                raise ValueError('''Data must be length 2 tuple or list of 
                                 type (data_x, data_y)''')

            # Run encoder on x and y
            z_params_enc, z = self.encoder([data_x, data_y, samples])
            
            # Run prior on x
            z_params_pri, _ = self.prior([data_x, samples])
            
            # Bundle latent parameters to pass to loss function
            z_params = tf.concat([z_params_pri, z_params_enc], axis=-1)
            
            # run decoder on data_x and z where z is sampled from encoder
            y_params, y = self.decoder([data_x, z, samples])
            
            # Add Kullback-Leibler loss and metric if using add_loss API
            if self.kl_loss:
                kl_loss = self.kl_loss(data_y, z_params)
                
                self.add_loss(self.beta * kl_loss)
                self.add_metric(kl_loss, aggregation='mean', name='kl_loss')
                
            # Add reconstruction loss and metric if using add_loss API
            if self.r_loss:
                r_loss = self.r_loss(data_y, y_params)
                
                self.add_loss(r_loss)
                self.add_metric(r_loss, aggregation='mean', name='r_loss')
            
            # Add beta metric
            self.add_metric(self.beta, aggregation='mean', name='beta')
            return {'y_params':y_params, 
                    'z_params':z_params, 
                    'y':y, 
                    'z':z}
        
        # Inference mode
        else:
            z_params, z = self.prior([data, samples])
            y_params, y = self.decoder([data, z, samples])

            if verbose == False:
                return y
            if verbose == True:
                return {'y_params':y_params, 
                        'z_params':z_params, 
                        'y':y, 
                        'z':z}
