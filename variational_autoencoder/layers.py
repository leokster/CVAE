import tensorflow as tf

from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.gen_math_ops import inv


class Sampling(tf.keras.layers.Layer):
    """Takes tensors of parameters as input and uses a tf.random.Generator
    object to sample element-wise from a desired distribution.
    Can also be passed parameters as scalars, along with a desired shape
    to output a tensor sampled from identical parameters.

    Inputs to constructor:
    dist - A string denoting a distribution from tf.random.Generator.
           Defaults to 'normal'."""

    def __init__(self, dist='normal'):
        super(Sampling, self).__init__()
        self.rn_gen = tf.random.Generator.from_non_deterministic_state()
        self.dist = getattr(self.rn_gen, dist)

    def call(self, params, shape=None):
        """Inputs:
        params - A list of parameters that has to be compatible with the
                 chosen distribution. Can be either a list of tensors, such
                 as [mu, sigma], where mu and sigma are float32 tensors,
                 or a list of scalars such as [0, 1], in which case the
                 desired output shape needs to be passed explicitly.
        shape -  A tensorshape object, optionally passed to sample from
                 scalar parameters."""

        if shape == None:
            shape = tf.shape(params[0])
        sample = self.dist(shape, *params)
        return sample


class StackNTimes(tf.keras.layers.Layer):
    """Expands dimension of input tensor and stacks n times on new axis.
    Works dynamically, with the amount of stacks n as a tensor with 
    data type int32, or a python or numpy integer type.

    Inputs to constructor:
    axis - The axis to stack on, ranging from (-old_dim - 1) to (old_dim)."""

    def __init__(self, axis=1):
        super(StackNTimes, self).__init__()
        self.axis = axis

    def call(self, inputs, n):
        """Inputs:
        inputs - The tensor to stack multiple times. Can be a tensor of any
                 rank or data type.
        n      - The amount of times to stack the tensor. Can be a tensor
                 or any python or numpy integer type. Will be coverted to
                 a tensor of int32 type internally."""
        old_dim = len(tf.shape(inputs))
        n = tf.cast(tf.expand_dims(n, axis=0), tf.int32)

        if (self.axis > old_dim) or (self.axis < -old_dim - 1):
            raise ValueError(f'Tried to stack on index {self.axis}' +
                             f' for tensor with {old_dim} dimensions.')

        elif (self.axis == 0) or (self.axis == -old_dim - 1):
            multiples = tf.concat([n,
                                   tf.ones([old_dim], dtype=tf.int32)],
                                  axis=0)

        elif (self.axis == -1):
            multiples = tf.concat([tf.ones([old_dim], dtype=tf.int32),
                                   n],
                                  axis=0)

        elif self.axis > 0:
            multiples = tf.concat([tf.ones([self.axis],
                                           dtype=tf.int32),
                                   n,
                                   tf.ones([old_dim - self.axis],
                                           dtype=tf.int32)],
                                  axis=0)

        elif self.axis < 0:
            multiples = tf.concat([tf.ones([old_dim + 1 + self.axis],
                                           dtype=tf.int32),
                                   n,
                                   tf.ones([-self.axis - 1],
                                           dtype=tf.int32)],
                                  axis=0)

        h = tf.expand_dims(inputs, axis=self.axis)
        outputs = tf.tile(h, multiples)
        return outputs


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
                shape=[second_last_dim, self.units, ],
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


class AddSamplingAxis(tf.keras.layers.Layer):
    """
    This Layer adds a sample axis as axis 1, dependant on the argument "sampling" it 
    then flattens the batch diemnsion and sampling dimension or it doesn't.
    """

    def __init__(self,
                 sampling='flattened',
                 **kwargs):
        """
        sampling can be either 'flattened', 'stacked' or False
        """
        super(AddSamplingAxis, self).__init__(**kwargs)

        self.axis = int(1)
        self.sampling = sampling
        if sampling in ('flattened', 'stacked'):
            self.stack_n_times = StackNTimes(axis=self.axis)
    
    def _add_samples(self, inputs, samples):
        if self.sampling == 'stacked':
            return self.stack_n_times(inputs, samples)

        if self.sampling == 'flattened':
            stacked = self.stack_n_times(inputs, samples)
            dims = tf.shape(stacked)
            dims_list = [dims[i] for i in range(len(dims))]
            return tf.reshape(stacked, (dims_list[0]*dims_list[1], *dims_list[2:]))

        if self.sampling is False:
            return inputs

    def call(self, inputs, samples, invert=False):
          samples = tf.cast(samples, tf.int32)

          if isinstance(inputs, list):
                return [self.call(element, samples, invert) for element in inputs]
    
          elif isinstance(inputs, tuple):
                return tuple((self.call(element, samples, invert) for element in inputs))
    
          elif isinstance(inputs, dict):
                return {key: self.call(val, samples, invert) for key, val in inputs.items()}
          
          elif not invert:
                return self._add_samples(inputs, samples)
          
          elif invert and self.sampling == 'flattened':
                dims = tf.shape(inputs)
                dims_list = [dims[i] for i in range(len(dims))]
                return tf.reshape(inputs, (-1, samples, *dims_list[1:]))
          
          elif invert and self.sampling in (False, 'stacked'):
                return inputs
          
          else:
                raise ValueError("inputs must be either a list, tuple, dict or tensor")