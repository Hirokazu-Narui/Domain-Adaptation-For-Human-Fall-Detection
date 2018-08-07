from codebase.models.extra_layers import leaky_relu, wndense, noise, wnconv2d, wnconv2d_transpose, scale_gradient
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorbayes.layers import dense, conv2d, conv2d_transpose, upsample, avg_pool, max_pool, batch_norm, instance_norm
from codebase.args import args

if 'gtsrb' in {args.src, args.trg}:
    Y = 43
elif 'cifar' in {args.src, args.trg}:
    Y = 9
else:
    Y = 2

dropout = tf.layers.dropout

def classifier(x, phase, enc_phase=1, trim=0, scope='class', reuse=None, internal_update=False, getter=None):
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        with arg_scope([leaky_relu], a=0.1), \
             arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=phase), \
             arg_scope([batch_norm], internal_update=internal_update):

            layout = [
                (instance_norm, (), {}),
                (conv2d, (64, [30,3], 1), dict(padding = 'VALID')),
                (conv2d, (64, [1,3], 1), dict(padding = 'VALID')),
                (conv2d, (64, [1,3], 1), dict(padding = 'VALID')),
                (conv2d, (64, [1,3], 1), dict(padding = 'VALID')),
                (max_pool, ([1,2], [1,2]), {}),
                (dropout, (), dict(training=phase)),
                (noise, (1,), dict(phase=phase)),
                (conv2d, (64, [1,3], 1), dict(padding = 'VALID')),
                (conv2d, (64, [1,3], 1), dict(padding = 'VALID')),
                (conv2d, (64, [1,3], 1), dict(padding = 'VALID')),
                (conv2d, (64, [1,3], 1), dict(padding = 'VALID')),
                (max_pool, ([1,2], [1,2]), {}),
                (dropout, (), dict(training=phase)),
                (noise, (1,), dict(phase=phase)),
                (conv2d, (64, [1,3], 1), dict(padding = 'VALID')),
                (conv2d, (64, [1,3], 1), dict(padding = 'VALID')),
                (conv2d, (64, [1,3], 1), dict(padding = 'VALID')),
                (conv2d, (64, [1,3], 1), dict(padding = 'VALID')),
#                (avg_pool, (), dict(global_pool=True)),
                (dense, (Y,), dict(activation=None))
            ]

            if enc_phase:
                start = 0
                end = len(layout) - trim
            else:
                start = len(layout) - trim
                end = len(layout)

            for i in xrange(start, end):
                with tf.variable_scope('l{:d}'.format(i)):
                    f, f_args, f_kwargs = layout[i]
                    x = f(x, *f_args, **f_kwargs)
                    print(i,x.shape)

    return x

def feature_discriminator(x, phase, C=1, reuse=None):
    with tf.variable_scope('disc/feat', reuse=reuse):
        with arg_scope([dense], activation=tf.nn.relu): # Switch to leaky?

            x = dense(x, 100)
            x = dense(x, C, activation=None)

    return x

def discriminator(x, phase, reuse=None):
    with tf.variable_scope('disc/gan', reuse=reuse):
        with arg_scope([wnconv2d, wndense], activation=leaky_relu):

            x = dropout(x, rate=0.2, training=phase)
            x = wnconv2d(x, 64, 3, 2)

            x = dropout(x, training=phase)
            x = wnconv2d(x, 128, 3, 2)

            x = dropout(x, training=phase)
            x = wnconv2d(x, 256, 3, 2)

            x = dropout(x, training=phase)
            x = wndense(x, 1024)

            x = dense(x, 1, activation=None, bn=False)

    return x

def encoder(x, y, phase, reuse=None):
    with tf.variable_scope('enc', reuse=reuse):
        with arg_scope([conv2d, dense], bn=True, phase=phase, activation=leaky_relu), \
             arg_scope([noise], phase=phase):

            # Ignore y
            x = conv2d(x, 64, 3, 2, bn=False)
            x = conv2d(x, 128, 3, 2)
            x = conv2d(x, 256, 3, 2)
            x = dense(x, 1024)

            m = dense(x, 100, activation=None)
            v = dense(x, 100, activation=tf.nn.softplus) + 1e-5

            return (m, v)

def generator(x, y, phase, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        with arg_scope([dense], bn=True, phase=phase, activation=tf.nn.relu), \
             arg_scope([conv2d_transpose], bn=True, phase=phase, activation=tf.nn.relu):

            if y is not None:
                x = tf.concat([x, y], 1)

            x = dense(x, 4 * 4 * 512)
            x = tf.reshape(x, [-1, 4, 4, 512])
            x = conv2d_transpose(x, 256, 5, 2)
            x = conv2d_transpose(x, 128, 5, 2)
            x = wnconv2d_transpose(x, 3, 5, 2, bn=False, activation=tf.nn.tanh, scale=True)

    return x
