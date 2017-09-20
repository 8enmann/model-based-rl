import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten


def sample(logits):
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)


class FcPolicy(object):
    """Fully connected NN. Expects input to be flat."""

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        ob_shape = (None, ob_space.shape[0] * nstack,)
        nact = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape)
        with tf.variable_scope("model", reuse=reuse):
            h1 = Dense(64, activation='selu', name='fc1')(X)
            h2 = Dense(64, activation='selu', name='fc2')(h1)
            h2 = Dense(64, activation='selu', name='fc2')(h2)
            pi = Dense(nact, name='pi')(h2)
            vf = Dense(1, name='v')(h2)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nh, nw, nc = ob_space.shape
        ob_shape = (None, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            x = Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', name='c1')(tf.cast(X, tf.float32)/255.)
            x = Conv2D(64, kernel_size=4, strides=2, activation='relu', name='c2')(x)
            x = Conv2D(64, kernel_size=3, strides=1, activation='relu', name='c3')(x)
            x = Flatten()(x)
            h5 = Dense(512, activation='relu', name='fc1')(x)
            pi = Dense(nact, name='pi')(h5)
            vf = Dense(1, name='v')(h5)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
