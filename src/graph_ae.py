from __future__ import division, print_function
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, add, Flatten, Dropout, Activation, BatchNormalization, Lambda, LeakyReLU, UpSampling1D, AveragePooling1D, Layer
from keras import backend as K
from keras.optimizers import Adam
from tensorflow import pad

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

    def output_block(layer, config):
        #outputs = Dense(len_classes, activation='softmax')(layer)
        kernel_size = 8
        layer = Conv1D(filters=config.filter_length,
               kernel_size=kernel_size,
               padding='same',
               strides=2,
               kernel_initializer='he_normal')(layer)
        layer = Activation(LeakyReLU(alpha=0.2))(layer)
        layer = ReflectionPadding1D()(layer)

        from keras.layers.wrappers import TimeDistributed
        outputs = TimeDistributed(Dense(2, Activation(LeakyReLU(alpha=0.2))))(layer)
        #outputs = Dense(1, activation=Activation(LeakyReLU(alpha=0.2)))(layer)

        model = Model(inputs=inputs, outputs = outputs)
        
        adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer= adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        model.summary()
        return model
 
    classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']
    len_classes = len(classes)

    inputs = Input(shape=(config.input_size, 1), name='input')
    layer = first_conv_block(inputs, config)
    return output_block(layer, config)


def decoder_model(config):

    def first_conv_block(inputs, config):

        kernel_size = 8
        s = 2

        layer = UpSampling1D(size=2)(inputs)
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

        '''
        layer = AveragePooling1D()(layer)

        layer = UpSampling1D(size=2)(layer)
        layer = Conv1D(filters=config.filter_length,
               kernel_size=kernel_size,
               padding='same',
               strides=s,
               kernel_initializer='he_normal',
               activation=LeakyReLU(alpha=0.2))(layer)
               
        layer = Conv1D(filters=config.filter_length,
            kernel_size=kernel_size,
            padding='same',
            strides=s,
            kernel_initializer='he_normal',
            activation=LeakyReLU(alpha=0.2))(layer)
        layer = AveragePooling1D()(layer)

        layer = Conv1D(filters=config.filter_length,
            kernel_size=kernel_size,
            padding='same',
            strides=s,
            kernel_initializer='he_normal',)(layer)
        layer = AveragePooling1D()(layer)
        '''
        return layer

    def output_block(layer, config):
        from keras.layers.wrappers import TimeDistributed
        outputs = TimeDistributed(Dense(1, LeakyReLU(alpha=0.2)))(layer)
        #outputs = TimeDistributed(Dense(len_classes, activation='softmax'))(layer)
        model = Model(inputs=inputs, outputs = outputs)
        
        adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer= adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        model.summary()
        return model
 
    classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']
    len_classes = len(classes)

    inputs = Input(shape=(config.input_size, 2), name='input')
    layer = first_conv_block(inputs, config)
    return output_block(layer, config)

def vae_model(config):

    encoder = encoder_model(config)
    decoder = decoder_model(config)
    inputs = Input(shape=(config.input_size, 1), name='input')
    outputs = decoder(encoder(inputs))
    model = Model(inputs, outputs, name='vae')

    from keras.losses import binary_crossentropy
    def kl_reconstruction_loss(true, pred):
        # Reconstruction loss
        reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * config.input_size
        '''
        # KL divergence loss
        kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(reconstruction_loss + kl_loss)
        '''
        return reconstruction_loss


    model.compile(optimizer='adam', loss = kl_reconstruction_loss)

    return model