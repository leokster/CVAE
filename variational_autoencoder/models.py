import tensorflow as tf

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, prior, **kwargs):
        self.reconstruction_weight = tf.Variable(min(kwargs.pop("reconstruction_weight", 0.5), 1)*1.0, trainable=False)
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

    def call(self, data, training=False, sample_size=1):
        #if in training mode
        if training:
            #unpack data
            data_x, data_y, = data

            #run encoder on data_x and data_y
            z_mean, z_log_var, z, = self.encoder([data_x, data_y])

            #run prior on data_x
            z_prior_mean, z_prior_log_var, zz = self.prior(data_x)

            #compute Kullback–Leibler divergence between prior and encoder
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - z_prior_log_var - tf.exp(-z_prior_log_var) * (
                    tf.exp(z_log_var) + tf.square(z_mean - z_prior_mean)))

            #run decoder on data_x and z where z is sampled from encoder
            reconstruction_mean, reconstruction_log_var, _ = self.decoder([data_x, z])

            #compute likelihood of the true data_y value under the decoder distribution
            reconstruction_loss = tf.reduce_mean(
                0.5 * tf.square(data_y - reconstruction_mean) * tf.exp(-reconstruction_log_var) +
                reconstruction_log_var
            )

            #compute weighted sum of the two losses
            total_loss = self.reconstruction_weight * reconstruction_loss + \
                         (1 - self.reconstruction_weight) * kl_loss

            #add loss for gradient computation
            self.add_loss(total_loss)

            #add metrices for training logs
            self.add_metric(reconstruction_loss, aggregation='mean', name='r_loss')
            self.add_metric(kl_loss, aggregation='mean', name='kl_loss')
            self.add_metric(self.reconstruction_weight, aggregation='mean', name='rec_weight')
            return reconstruction_mean, reconstruction_log_var, z_log_var, z_mean

        #if in inference mode
        else:
            _, _, z, = self.prior(data)
            _, _, y_sampled = self.decoder([data, z])
            return y_sampled
