import tensorflow as tf
import tensorbayes as tb
import numpy as np
from tensorbayes.layers import *
from tensorbayes.distributions import *
from tensorflow.contrib.framework import arg_scope, add_arg_scope

################
# Extra layers #
################
def normalize_vector(d, scope=None):
    with tf.name_scope(scope, 'norm_vec'):
        ndim = len(d.shape)
        output = d * tf.rsqrt(1e-6 + tf.reduce_sum(tf.square(d), axis=range(1, ndim), keep_dims=True))
    return output

@add_arg_scope
def basic_accuracy(a, b, scope=None):
    with tf.name_scope(scope, 'basic_acc'):
        a = tf.argmax(a, 1)
        b = tf.argmax(b, 1)
        eq = tf.cast(tf.equal(a, b), 'float32')
        output = tf.reduce_mean(eq)
    return output

@add_arg_scope
def scale_gradient(x, scale, scope=None, reuse=None):
    with tf.name_scope('scale_grad'):
        output = (1 - scale) * tf.stop_gradient(x) + scale * x
    return output

@add_arg_scope
def noise(x, std, phase, scope=None, reuse=None):
    with tf.name_scope(scope, 'noise'):
        eps = tf.random_normal(tf.shape(x), 0.0, std)
        output = tf.where(phase, x + eps, x)
    return output

@add_arg_scope
def leaky_relu(x, a=0.2, name=None):
    with tf.name_scope(name, 'leaky_relu'):
        return tf.maximum(x, a * x)

@add_arg_scope
def wndense(x,
            num_outputs,
            scope=None,
            activation=None,
            reuse=None,
            scale=False,
            shift=True,
            bn=False,
            post_bn=False,
            phase=None):

    with tf.variable_scope(scope, 'wndense', reuse=reuse):
        # convert x to 2-D tensor
        dim = np.prod(x._shape_as_list()[1:])
        x = tf.reshape(x, [-1, dim])
        weights_shape = (x.get_shape().dims[-1], num_outputs)

        # dense layer
        weights = tf.get_variable('weights', weights_shape,
                                  initializer=variance_scaling_initializer())
        weights_norm = tf.sqrt(1e-6 + tf.reduce_sum(tf.square(weights), axis=0, keep_dims=True))
        weights = weights / weights_norm
        if scale:
            theta = tf.get_variable('theta', (1, num_outputs),
                                    initializer=variance_scaling_initializer())
            weights = weights * theta

        output = tf.matmul(x, weights)
        if shift:
            biases = tf.get_variable('biases', [num_outputs],
                                     initializer=tf.zeros_initializer)
            output = output + biases
        if bn: output = batch_norm(output, phase, scope='bn')
        if activation: output = activation(output)
        if post_bn: output = batch_norm(output, phase, scope='post_bn')
    return output

@add_arg_scope
def conv2d(x,
           num_outputs,
           kernel_size,
           strides,
           padding='SAME',
           activation=None,
           bn=False,
           post_bn=False,
           phase=None,
           scope=None,
           reuse=None):
    # Convert int to list
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    strides = [strides] * 2 if isinstance(strides, int) else strides

    # Convert list to valid list
    kernel_size = list(kernel_size) + [x.get_shape().dims[-1], num_outputs]
    strides = [1] + list(strides) + [1]

    # Conv operation
    with tf.variable_scope(scope, 'conv2d', reuse=reuse):
        if padding == 'REFLECT':
            if kernel_size[0] > strides[1]:
                padding = [0, kernel_size[0] / 2, kernel_size[1] / 2, 0]
                padding = [[v, v] for v in padding]
                x = tf.pad(x, padding, mode='REFLECT')
            padding = 'VALID'

        kernel = tf.get_variable('weights', kernel_size,
                                 initializer=variance_scaling_initializer())
        biases = tf.get_variable('biases', [num_outputs],
                                 initializer=tf.zeros_initializer)
        output = tf.nn.conv2d(x, kernel, strides, padding, name='conv2d')
        output += biases
        if bn: output = batch_norm(output, phase, scope='bn')
        if activation: output = activation(output)
        if post_bn: output = batch_norm(output, phase, scope='post_bn')
    return output

@add_arg_scope
def wnconv2d(x,
             num_outputs,
             kernel_size,
             strides,
             padding='SAME',
             activation=None,
             scale=False,
             shift=True,
             bn=False,
             post_bn=False,
             phase=None,
             scope=None,
             reuse=None):
    # Convert int to list
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    strides = [strides] * 2 if isinstance(strides, int) else strides

    # Convert list to valid list
    kernel_size = list(kernel_size) + [x.get_shape().dims[-1], num_outputs]
    strides = [1] + list(strides) + [1]

    with tf.variable_scope(scope, 'wnconv2d', reuse=reuse):
        kernel = tf.get_variable('weights', kernel_size,
                                 initializer=variance_scaling_initializer())
        kernel_norm = tf.sqrt(1e-6 + tf.reduce_sum(tf.square(kernel), axis=[0,1,2], keep_dims=True))
        kernel /= kernel_norm
        if scale:
            theta = tf.get_variable('theta', (1, 1, 1, num_outputs),
                                    initializer=variance_scaling_initializer())
            kernel *= theta
        output = tf.nn.conv2d(x, kernel, strides, padding, name='conv2d')
        if shift:
            biases = tf.get_variable('biases', [num_outputs],
                                     initializer=tf.zeros_initializer)
            output += biases
        if bn: output = batch_norm(output, phase, scope='bn')
        if activation: output = activation(output)
        if post_bn: output = batch_norm(output, phase, scope='post_bn')
        return output

@add_arg_scope
def wnconv2d_transpose(x,
                       num_outputs,
                       kernel_size,
                       strides,
                       padding='SAME',
                       output_shape=None,
                       output_like=None,
                       activation=None,
                       scale=False,
                       shift=True,
                       bn=False,
                       post_bn=False,
                       phase=None,
                       scope=None,
                       reuse=None):
    # Convert int to list
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    strides = [strides] * 2 if isinstance(strides, int) else strides

    # Convert list to valid list
    kernel_size = list(kernel_size) + [num_outputs, x.get_shape().dims[-1]]
    strides = [1] + list(strides) + [1]

    # Get output shape both as tensor obj and as list
    if output_shape:
        bs = tf.shape(x)[0]
        _output_shape = tf.stack([bs] + output_shape[1:])

    elif output_like:
        _output_shape = tf.shape(output_like)
        output_shape = output_like.get_shape()

    else:
        assert padding == 'SAME', "Shape inference only applicable with padding is SAME"
        bs, h, w, c = x._shape_as_list()
        bs_tf = tf.shape(x)[0]
        _output_shape = tf.stack([bs_tf, strides[1] * h, strides[2] * w, num_outputs])
        output_shape = [bs, strides[1] * h, strides[2] * w, num_outputs]

    # Transposed conv operation
    with tf.variable_scope(scope, 'conv2d', reuse=reuse):
        kernel = tf.get_variable('weights', kernel_size,
                                 initializer=variance_scaling_initializer())
        kernel_norm = tf.sqrt(1e-6 + tf.reduce_sum(tf.square(kernel), axis=[0,1,3], keep_dims=True))
        kernel /= kernel_norm

        if scale:
            theta = tf.get_variable('theta', (1, 1, num_outputs, 1),
                                    initializer=variance_scaling_initializer())
            kernel *= theta

        output = tf.nn.conv2d_transpose(x, kernel, _output_shape, strides,
                                        padding, name='conv2d_transpose')

        if shift:
            biases = tf.get_variable('biases', [num_outputs],
                                     initializer=tf.zeros_initializer)
            output += biases

        output.set_shape(output_shape)
        if bn: output = batch_norm(output, phase, scope='bn')
        if activation: output = activation(output)
        if post_bn: output = batch_norm(output, phase, scope='post_bn')
    return output

def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    output = tf.nn.softmax( y / temperature)
    return output

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
