import tensorflow as tf

from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import dtypes


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    
    def __init__(self, dist='normal'):
        super(Sampling, self).__init__()
        self.dist = getattr(tf.random, dist)

    def call(self, params):
        shape = params[0].shape
        sample = self.dist(shape, *params)
        return sample

class Dense2d(tf.keras.layers.Layer):
  """Extension of the regular Dense layer to a 3d (and higher) input
  Tensor. Different from the regular Dense layer, weights will not
  be shared for the second last dimension. A typical application is,
  if you want to train multiple mini machines in parallel. This layer is
  equivalent to build multiple independent Dense layers in parallel.
  `Dense2d` implements the operation:
  `output = activation(matvec(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights tensor
  created by the layer, and `bias` is a bias matrix created by the layer
  (only applicable if `use_bias` is `True`).
  For example, if input has dimensions `(batch_size, d0, d1, d2)`,
  then we create a `kernel` with shape `(d1, units, d2)`, and the `kernel` operates
  along axes 2 and 3 of the `input`, on every sub-tensor of shape `(1, 1, d1, d2)`
  (there are `batch_size * d0` such sub-tensors).
  The output in this case will have shape `(batch_size, d0, d1, units)`.
  Besides, layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).
  Example:
  >>> # Create a `Sequential` model and add a Dense2d layer as the first layer.
  >>> model = tf.keras.models.Sequential()
  >>> model.add(tf.keras.Input(shape=(100, 16)))
  >>> model.add(tf.keras.layers.Dense2d(32, activation='relu'))
  >>> # Now the model will take as input arrays of shape (None, 100, 16)
  >>> # and output arrays of shape (None, 100, 32).
  >>> # Note that after the first layer, you don't need to specify
  >>> # the size of the input anymore:
  >>> model.add(tf.keras.layers.Dense2d(32))
  >>> model.output_shape
  (None, 100, 32)
  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
  Input shape:
    N-D tensor with shape: `(batch_size, ..., n_machines, input_dim)`.
    The most common situation would be
    a 3D input with shape `(batch_size, n_machines, input_dim)`.
  Output shape:
    N-D tensor with shape: `(batch_size, ..., n_machines, units)`.
    For instance, for a 3D input with shape `(batch_size, n_machines, input_dim)`,
    the output would have shape `(batch_size, n_machines, units)`.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Dense2d, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)

    self.units = int(units) if not isinstance(units, int) else units
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.input_spec = InputSpec(min_ndim=3)
    self.supports_masking = True

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))

    input_shape = tensor_shape.TensorShape(input_shape)
    second_last_dim = tensor_shape.dimension_value(input_shape[-2])
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `Dense2d` '
                       'should be defined. Found `None`.')

    if second_last_dim is None:
      raise ValueError('The second last dimension of the inputs to `Dense2d` '
                       'should be defined. Found `None`.')

    self.input_spec = InputSpec(min_ndim=3, axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[second_last_dim, self.units, last_dim],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[second_last_dim, self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
      h = tf.linalg.matvec(self.kernel, inputs) + self.bias
      if self.activation is None:
          return h
      else:
          return self.activation(h)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = super(Dense2d, self).get_config()
    config.update({
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    })
    return config

