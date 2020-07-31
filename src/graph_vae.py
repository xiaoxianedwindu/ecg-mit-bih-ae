from __future__ import division, print_function
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, add, Flatten, Dropout, Activation, BatchNormalization, Lambda, LeakyReLU, UpSampling1D, AveragePooling1D, Layer, Reshape
from keras import backend as K
from keras.optimizers import Adam
from tensorflow import pad

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()



'''
  1D Reflection Padding
  Attributes:
    - padding: (padding_left, padding_right) tuple
'''
class ReflectionPadding1D(Layer):
    def __init__(self, padding=(64, 64), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding1D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[1] + self.padding[0] + self.padding[1]

    def call(self, input_tensor, mask=None):
        padding_left, padding_right = self.padding
        return pad(input_tensor,  [[0, 0], [padding_left, padding_right], [0, 0]], mode='REFLECT')

    def get_config(self):
        config = super(ReflectionPadding1D, self).get_config()
        #print(config)
        return config

class ReflectionPadding1D_decode(Layer):
    def __init__(self, padding=(128, 128), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding1D_decode, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[1] + self.padding[0] + self.padding[1]

    def call(self, input_tensor, mask=None):
        padding_left, padding_right = self.padding
        return pad(input_tensor,  [[0, 0], [padding_left, padding_right], [0, 0]], mode='REFLECT')

    def get_config(self):
        config = super(ReflectionPadding1D_decode, self).get_config()
        #print(config)
        return config

def sample_z(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
    """
    mu, sigma = args
    batch     = K.shape(mu)[0]
    dim       = K.int_shape(mu)[1]
    eps       = K.random_normal(shape=(batch, dim))
    print('================================')
    print(mu)
    print(sigma)
    print(batch)
    print(dim)
    print(eps.shape)
    print(K.exp(sigma / 2).shape)
    print('================================')
    return mu + K.exp(sigma / 2) * eps

def vae_model(config):
    def encoder_model(config):
        def first_conv_block(inputs, config):

            kernel_size =16
            s = 2

            layer = Conv1D(filters=config.filter_length,
                kernel_size=kernel_size,
                padding='same',
                strides=2,
                kernel_initializer='he_normal',
                activation=LeakyReLU(alpha=0.2))(inputs)
            layer = ReflectionPadding1D()(layer)

            layer = Conv1D(filters=config.filter_length,
                kernel_size=kernel_size,
                padding='same',
                strides=2,
                kernel_initializer='he_normal')(layer)
            layer = Activation(LeakyReLU(alpha=0.2))(layer)
            layer = ReflectionPadding1D()(layer)

            layer = Conv1D(filters=config.filter_length,
                kernel_size=kernel_size,
                padding='same',
                strides=2,
                kernel_initializer='he_normal')(layer)
            layer = Activation(LeakyReLU(alpha=0.2))(layer)
            layer = ReflectionPadding1D()(layer)

            layer = Conv1D(filters=config.filter_length,
                kernel_size=kernel_size,
                padding='same',
                strides=2,
                kernel_initializer='he_normal')(layer)
            layer = Activation(LeakyReLU(alpha=0.2))(layer)
            layer = ReflectionPadding1D()(layer)

            return layer

        en_inputs = Input(shape=(config.input_size, 1), name='encoder_input')
        layer = first_conv_block(en_inputs, config)
        kernel_size = 8
        layer = Conv1D(filters=config.filter_length,
            kernel_size=kernel_size,
            padding='same',
            strides=2,
            kernel_initializer='he_normal')(layer)
        layer = Activation(LeakyReLU(alpha=0.2))(layer)
        layer = ReflectionPadding1D()(layer)

        from keras.layers.wrappers import TimeDistributed
        #outputs = TimeDistributed(Dense(1, Activation(LeakyReLU(alpha=0.2))))(layer)
        outputs = Flatten()(layer)
        outputs = Dense(20, activation=Activation(LeakyReLU(alpha=0.2)))(outputs)
        outputs = BatchNormalization()(outputs)
        mu = (Dense(2, name='latent_mu'))(outputs)
        sigma = (Dense(2, name='latent_sigma'))(outputs)

        '''
        mu: mean values of encoded input
        sigma: stddev of encoded input
        '''
        z = Lambda(sample_z, output_shape=(2, ), name='z')([mu, sigma])
        #conv_shape = K.int_shape(layer)
        encoder = Model(inputs=en_inputs, outputs = [mu, sigma, z], name = 'Encoder')
        encoder.summary() 
        return encoder, mu, sigma

    def decoder_model(config):

        def first_conv_block(inputs, config):

            layer = Dense(config.input_size* 32, activation =LeakyReLU(alpha=0.2))(inputs)
            layer = Reshape((config.input_size, 1))(layer)

            kernel_size = 8
            s = 2

            layer = UpSampling1D(size=2)(layer)
            layer = Conv1D(filters=config.filter_length,
                kernel_size=kernel_size,
                padding='same',
                strides=s,
                kernel_initializer='he_normal',
                )(layer)
            layer = ReflectionPadding1D_decode()(layer)
            layer = AveragePooling1D()(layer)
            layer = Activation(LeakyReLU(alpha=0.2))(layer)

            layer = UpSampling1D(size=2)(layer)
            layer = Conv1D(filters=config.filter_length,
                kernel_size=kernel_size,
                padding='same',
                strides=s,
                kernel_initializer='he_normal',
                )(layer)
            layer = ReflectionPadding1D_decode()(layer)
            layer = AveragePooling1D()(layer)
            layer = Activation(LeakyReLU(alpha=0.2))(layer)

            layer = UpSampling1D(size=2)(layer)
            layer = Conv1D(filters=config.filter_length,
                kernel_size=kernel_size,
                padding='same',
                strides=s,
                kernel_initializer='he_normal',
                )(layer)
            layer = ReflectionPadding1D_decode()(layer)
            layer = AveragePooling1D()(layer)
            layer = Activation(LeakyReLU(alpha=0.2))(layer)

            layer = UpSampling1D(size=2)(layer)
            layer = Conv1D(filters=config.filter_length,
                kernel_size=kernel_size,
                padding='same',
                strides=s,
                kernel_initializer='he_normal',
                )(layer)
            layer = ReflectionPadding1D_decode()(layer)
            layer = AveragePooling1D()(layer)
            layer = Activation(LeakyReLU(alpha=0.2))(layer)

            layer = UpSampling1D(size=2)(layer)
            layer = Conv1D(filters=config.filter_length,
                kernel_size=kernel_size,
                padding='same',
                strides=s,
                kernel_initializer='he_normal',
                )(layer)

            return layer

        def output_block(layer, config):
            from keras.layers.wrappers import TimeDistributed
            outputs = TimeDistributed(Dense(1, LeakyReLU(alpha=0.2)))(layer)
            #outputs = TimeDistributed(Dense(len_classes, activation='softmax'))(layer)
            model = Model(inputs=inputs, outputs = outputs, name='Decoder')
            model.summary()
            return model


        inputs = Input(shape=(2, ), name='decoder_input')
        layer = first_conv_block(inputs, config)
        return output_block(layer, config)

    encoder, mu, sigma = encoder_model(config)
    decoder = decoder_model(config)
    inputs = Input(shape=(config.input_size, 1), name='vae_input')
    #encoder_m, mu, sigma = 
    outputs = decoder(encoder(inputs))
    model = Model(inputs, outputs, name='vae')
    model.summary()

    from keras.losses import binary_crossentropy
    def kl_reconstruction_loss(true, pred):
        # Reconstruction loss
        reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * config.input_size
        # KL divergence loss
        kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(reconstruction_loss + kl_loss)

    model.compile(optimizer='adam', loss = kl_reconstruction_loss)

    return model