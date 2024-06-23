import tensorflow as tf
import numpy as np



class PowerNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self,A,n, epsilon=1e-5, momentum=0.99, **kwargs):
        super(PowerNormalizationLayer, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.momentum = momentum
        self.A = A
        self.n = n
        
    def build(self, input_shape):
        # Asegúrate de que input_shape es una tupla
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) == 1:
            shape = (input_shape[-1],)
        else:
            shape = input_shape[-1:]



        self.moving_mean = self.add_weight(name='moving_mean',
                                           shape=shape,
                                           initializer='zeros',
                                           trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
                                               shape=shape,
                                               initializer='ones',
                                               trainable=False)

    def call(self, inputs, training=False):
        if training:
            batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0])
            mean = batch_mean
            variance = batch_variance
            self.moving_mean.assign(self.moving_mean * self.momentum + mean * (1 - self.momentum))
            self.moving_variance.assign(self.moving_variance * self.momentum + variance * (1 - self.momentum))
        else:
            mean = self.moving_mean
            variance = self.moving_variance

        normalized_inputs = np.sqrt(self.A) * ((inputs - mean) / tf.sqrt(2 * (variance + self.epsilon)))
        E = tf.reduce_sum(tf.square(normalized_inputs), axis=1, keepdims=True)
        P = tf.reduce_mean(E)
        return normalized_inputs

    def get_config(self):
        config = super(PowerNormalizationLayer, self).get_config()
        config.update({
            'A': self.A,
            'n': self.n,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)