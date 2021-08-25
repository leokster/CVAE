from operator import invert
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
from .layers import StackNTimes

if int(tf.__version__.replace(".", "")) < 240:
    from tensorflow.python.keras.engine.training import _minimize


def _get_input_shape(layer):
    return layer.layers[0].output_shape


def _get_output_len(layer):
    return len(layer.output)

    
class VAE(tf.keras.Model):
    def __init__(self, encoder_zero, decoder_zero, prior_zero,
                 encoder=None, decoder=None, prior=None, 
                 kl_loss=None, r_loss=None, latent_dim=None, 
                 training_mode_samples=1, 
                 inference_samples_train=10,
                 inference_samples_test=10, 
                 inference_samples_predict=1000,
                 do_sampling=False,
                 pass_samples_to_model=True,
                 **kwargs):
        """
        Initialize the Conditional Variational Autoencoder model as subclass of tf.keras.Model.
        In the following X is a tensor in the domain space and y is a tensor in the target space
        (including) batch sizes.

        Input parameters:
        :param encoder: tf.keras.Model taking a list of three tensors as input [X, y, samples] and outputs
        a list of 3 tensors [Z_mean, Z_logvar, Z_smpl] in the latent space, where Z_mean is the mean
        of a Gaussian, Z_logvar the logvariance of the Gaussian and Z_smpl one sample of the corresponding
        distribution (can be realized with the Sampling layer).
        :param prior: tf.keras.Model taking a list of tensors [X, samples] as input and outputs a list of 3 tensors as the encoder
        does
        :param decoder: tf.keras.Model taking a list of three tensors [X, Z, samples]
        where Z are samples of the latent space and maps it to the target
        space y.
        :param do_sampling: can be either "flattened", "stacked" or False. If "flattened", "stacked" the engine will automatically
        add copies of the input either in the batch dimension (for flattened) or as an additional dimension (for stacked) before
        passing to the individual models. If set to False, it will not create any samples automatically and they have to be created
        manually in the individual submodels (prior, encoder, decoder). The call method returns the sampling axis (if not False) anyway
        separate again, no matter whether do_sampling is "flattened" or "stacked".
        
        Call method returns:
        In training mode  - A dict of {'y_params':y_params, 'z_params': z_params,
                                       'y':y, 'z':z}
        In inference mode - The same dict if verbose is set to true,
                            otherwise simply outputs the sampled y.

        Example:

        def make_prior(latent_dim=2):
            sample_input = tf.keras.Input(shape=[], batch_size=1)
            samples = tf.squeeze(sample_input)
            
            prior_input = tf.keras.layers.Input(13)
            # add some layers
            # x = tf.keras.layers.Dense(64)
            z_mu = tf.keras.layers.Dense(latent_dim)(x)
            z_logvar = tf.keras.layers.Dense(latent_dim)(x)
            z_smpl = Sampling()([z_mu, z_logvar])
            return tf.keras.models.Model([prior_input, sample_input], 
                                         [z_mu, z_logvar, z_smpl])

        def make_encoder(latent_dim=2):
            sample_input = tf.keras.Input(shape=[], batch_size=1)
            samples = tf.squeeze(sample_input)
            
            x_input = tf.keras.layers.Input(13)
            y_input = tf.keras.layers.Input(1)
            x = tf.keras.layers.concatenate([x_input, y_input])
            # add some layers
            # x = tf.keras.layers.Dense(64)
            z_mu = tf.keras.layers.Dense(latent_dim)(x)
            z_logvar = tf.keras.layers.Dense(latent_dim)(x)
            z_smpl = Sampling()([z_mu, z_logvar])
            return tf.keras.models.Model([x_input, y_input, sample_input], 
                                         [z_mu, z_logvar, z_smpl])

        def make_decoder(latent_dim=2):
            sample_input = tf.keras.Input(shape=[], batch_size=1)
            samples = tf.squeeze(sample_input)
    
            x_input = tf.keras.layers.Input(13)
            z_input = tf.keras.layers.Input(latent_dim)
            x = tf.keras.layers.concatenate([x_input, z_input])
            # add some layers
            # x = tf.keras.layers.Dense(64)
            y_mu = tf.keras.layers.Dense(1)(x)
            y_logvar = tf.keras.layers.Dense(1)(x)
            y_smpl = Sampling()([y_mu, y_logvar])
            y_params = tf.keras.layers.concatenate([y_mu, y_logvar], axis=0)
            return tf.keras.models.Model([x_input, z_input, sample_input],
                                         [y_params, y_smpl])
        """
        
        self.beta = tf.Variable(kwargs.pop("beta", 1) * 1.0, trainable=False)
        super(VAE, self).__init__(**kwargs)
        self.encoder_zero = encoder_zero
        self.decoder_zero = decoder_zero
        self.prior_zero = prior_zero
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
        self.stack = StackNTimes(axis=1)

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
            y_pred_inference = self((x, y[:-1]), training=False, 
                                    samples=self.inference_samples_train,
                                    verbose=True)
            self.compiled_metrics.update_state(y[-1], y_pred_inference, 
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
        y_pred = self((x,y), training=True,
                      samples=self.training_mode_samples)
        
        # Updates stateful loss metrics.
        self.compiled_loss(y, y_pred, sample_weight, 
                           regularization_losses=self.losses)
        
        # Run in inference mode for other metrics
        if self.compiled_metrics._metrics is not None:
            y_pred_inference = self((x, y[:-1]), training=False, 
                                    samples=self.inference_samples_test, 
                                    verbose=True)
            self.compiled_metrics.update_state(y[-1], y_pred_inference, 
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
        x, y, _ = data_adapter.unpack_x_y_sample_weight(data)
        
        return self((x, y[:-1]), training=False, 
                    samples=self.inference_samples_predict)

    def build(self, input_shape):      
        # Instantiate networks if passed as subclasses
        if isinstance(self.encoder_zero, type):
            self.encoder_zero = self.encoder_zero(latent_dim=self.latent_dim)
        if isinstance(self.prior_zero, type):
            self.prior_zero = self.prior_zero(latent_dim=self.latent_dim)
        if isinstance(self.decoder_zero, type):
            assert tf.rank(input_shape)==3, (
                '''Cannot infer decoder output shape from x data only.
                Please call model on (x,y) in training mode to build'''
                )
            self.output_dim = input_shape[-1][-1][-1]
            self.decoder_zero = self.decoder_zero(output_dim=self.output_dim)
        if isinstance(self.encoder, type):
            self.encoder = self.encoder(latent_dim=self.latent_dim,
                                        encoder_zero=self.encoder_zero)
        if isinstance(self.prior, type):
            self.prior = self.prior(latent_dim=self.latent_dim,
                                    prior_zero=self.prior_zero)
        if isinstance(self.decoder, type):
            assert tf.rank(input_shape)==3, (
                '''Cannot infer decoder output shape from x data only.
                Please call model on (x,y) in training mode to build'''
                )
            self.output_dim = input_shape[-1][-1][-1]
            self.decoder = self.decoder(output_dim=self.output_dim,
                                        decoder_zero=self.decoder_zero)
            
        # Instantiate losses if passed as subclasses
        if isinstance(self.kl_loss, type):
            self.kl_loss = self.kl_loss(normalize=self.output_dim)
        if isinstance(self.r_loss, type):
            self.r_loss = self.r_loss(normalize=self.output_dim)

    def call(self, data, training=False, samples=1, verbose=False):
        samples = tf.cast(samples, tf.int32)
               
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

            # Split off data for zeroth timestep
            data_x_zero = self.stack(data_x[0], samples)
            data_y_zero = self.stack(data_y[0], samples)

            # Keep data of other timesteps for iteration
            data_x_rest = data_x[1:]
            data_y_rest = data_y[1:]

            # Run zeroth timestep
            z_params_enc_zero, z_zero = self.encoder_zero([data_x_zero, 
                                                           data_y_zero])
            
            z_params_pri_zero, _ = self.prior_zero([data_x_zero])
            
            y_params_zero, y_zero = self.decoder_zero([data_x_zero, 
                                                       z_zero])
            
            # Initialize lists for other results of other timesteps
            z_params_enc = [z_params_enc_zero]
            z = [z_zero]
            
            z_params_pri = [z_params_pri_zero]
            
            y_params = [y_params_zero]
            y = [y_zero]
            
            # Initialize previous values of y and z
            y_p = data_y_zero
            z_p = z_zero
            
            # Run other timesteps
            for data_x_t, data_y_t in zip(data_x_rest, data_y_rest):
                
                data_x_t = self.stack(data_x_t, samples)
                data_y_t = self.stack(data_y_t, samples)
                
                z_params_enc_t, z_t = self.encoder([data_x_t, 
                                                    data_y_t,
                                                    z_p])
            
                z_params_pri_t, _ = self.prior([data_x_t,
                                                z_p])
            
                y_params_t, y_t = self.decoder([data_x_t, 
                                                z_t,
                                                y_p])
                
                # Append results to lists
                z_params_enc.append(z_params_enc_t)
                z.append(z_t)
                
                z_params_pri.append(z_params_pri_t)
                
                y_params.append(y_params_t)
                y.append(y_t)
                
                # Overwrite previous values of y and z
                y_p = data_y_t
                z_p = z_t

            # Concatenate for loss computation
            z_params_enc = tf.stack(z_params_enc, axis=-2)
            z_params_pri = tf.stack(z_params_pri, axis=-2)
            y_params = tf.stack(y_params, axis=-2)
            
            z = tf.stack(z, axis=-1)
            y = tf.stack(y, axis=-1)
            
            z_params = tf.concat([z_params_pri, z_params_enc], axis=-1)

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
            
            result = {'y_params':y_params, 
                      'z_params':z_params, 
                      'y':y, 
                      'z':z}
            
            return result
        
        # Inference mode
        else:
                     
            # Unpack data
            if isinstance(data, (list, tuple)):
                if len(data) == 2:
                    data_x, data_y = data
                elif len(data) == 1:
                    data_x = data[0]
                    data_y = None
                else:
                    raise ValueError('''Data must be length 2 tuple or list of 
                                     type (data_x, data_y) or length 1 tuple
                                     or list of type (data_x)''')
            else:
                raise ValueError('''Data must be length 2 tuple or list of 
                                 type (data_x, data_y) or length 1 tuple
                                 or list of type (data_x)''')
            
            # Run in bootstrap mode if no y passed
            if data_y is None:
            
                z_params, z = self.prior_zero([data_x])
            
                y_params, y = self.decoder_zero([data_x, z])
                
            # Run in recurrent mode otherwise:
            else: 
                
                # Split off x for the timestep to be predicted
                data_x_last = self.stack(data_x[-1], samples)
                
                # Split off data for zeroth timestep
                data_x_zero = self.stack(data_x[0], samples)
                data_y_zero = self.stack(data_y[0], samples)
                
                # Keep data of other timesteps for iteration
                data_x_rest = data_x[1:-1]
                data_y_rest = data_y[1:]
                
                # Run zeroth timestep
                _, z_zero = self.encoder_zero([data_x_zero, data_y_zero])
                
                # Initialize previous values of z
                z_p = z_zero
                
                # Run middle timesteps
                for data_x_t, data_y_t in zip(data_x_rest, data_y_rest):
                
                    data_x_t = self.stack(data_x_t, samples)
                    data_y_t = self.stack(data_y_t, samples)
                
                    _, z_p = self.encoder([data_x_t, data_y_t, z_p])
                    
                # Run last timestep
                y_p = self.stack(data_y[-1], samples)
                
                z_params, z = self.prior([data_x_last, z_p])
                
                y_params, y = self.decoder([data_x_last, z, y_p])
            
            # Return results
            if verbose == False:
                
                return y
            
            if verbose == True:
                result =  {'y_params':y_params, 
                           'z_params':z_params, 
                           'y':y, 
                           'z':z}
                
                return result
