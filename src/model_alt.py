from __future__ import division, print_function
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, add, Flatten, Dropout,MaxPooling1D, Activation, BatchNormalization, Lambda, AveragePooling1D
from keras import backend as K
from keras.optimizers import Adam

def ECG_model(config):
    """ 
    https://www.sciencedirect.com/science/article/pii/S0022073619304170#s0015
    """
    def first_conv_block(inputs, config):
        layer = Conv1D(filters=32,
               kernel_size=5,
               padding='same',
               strides=2,
               kernel_initializer='he_normal',
               activation='relu')(inputs)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)


        layer = MaxPooling1D(pool_size=3,
                      strides=2,
                      padding='same'
                      )(layer)

        #return add([shortcut, layer])
        return layer

    def res_block_1(layer, config):

        layer =  Conv1D(filters=config.filter_length,
               kernel_size=1,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer =  Conv1D(filters=config.filter_length,
               kernel_size=3,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer =  Conv1D(filters=128,
                        kernel_size=1,
                        padding='same',
                        strides=1,
                        kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)

        shortcut =  Conv1D(filters=128,
                        kernel_size=1,
                        padding='same',
                        strides=2,
                        kernel_initializer='he_normal',
                        activation= 'relu')(layer)
        shortcut = BatchNormalization()(layer)

        return add([shortcut, layer])
    def res_block_2(layer, config):

        layer =  Conv1D(filters=64,
               kernel_size=1,
               padding='same',
               strides=2,
               kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer =  Conv1D(filters=64,
               kernel_size=3,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer =  Conv1D(filters=256,
                        kernel_size=1,
                        padding='same',
                        strides=1,
                        kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)

        shortcut =  Conv1D(filters=256,
                        kernel_size=3,
                        padding='same',
                        strides=2,
                        kernel_initializer='he_normal',
                        activation='relu')(layer)
        shortcut = BatchNormalization()(layer)

        return add([shortcut, layer])
    def res_block_3(layer, config):

        layer =  Conv1D(filters=128,
               kernel_size=1,
               padding='same',
               strides=2,
               kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer =  Conv1D(filters=128,
               kernel_size=3,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer =  Conv1D(filters=512,
                        kernel_size=1,
                        padding='same',
                        strides=1,
                        kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)

        shortcut =  Conv1D(filters=512,
                        kernel_size=3,
                        padding='same',
                        strides=2,
                        kernel_initializer='he_normal',
                        activation='relu')(layer)
        shortcut = BatchNormalization()(layer)

        return add([shortcut, layer])
    def res_block_4(layer, config):

        layer =  Conv1D(filters=256,
               kernel_size=1,
               padding='same',
               strides=2,
               kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer =  Conv1D(filters=256,
               kernel_size=3,
               padding='same',
               strides=1,
               kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer =  Conv1D(filters=1024,
                        kernel_size=1,
                        padding='same',
                        strides=1,
                        kernel_initializer='he_normal')(layer)
        layer = BatchNormalization()(layer)

        shortcut =  Conv1D(filters=1024,
                        kernel_size=3,
                        padding='same',
                        strides=2,
                        kernel_initializer='he_normal',
                        activation='relu')(layer)
        shortcut = BatchNormalization()(layer)

        return add([shortcut, layer])

    def output_block(layer, config):
        from keras.layers.wrappers import TimeDistributed
        layer = AveragePooling1D(pool_size = 3)(layer)
        layer = Flatten()(layer)
        layer = Activation('tanh')(layer)
        #layer = Activation('softmax')(layer)
        outputs = Dense(len_classes, activation='softmax')(layer)
        model = Model(inputs=inputs, outputs=outputs)
        
        adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer= adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        model.summary()
        return model

    classes = ['A', 'E', 'j', 'L', 'N', '/', 'R', 'V']#['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S'] are too few or not in the trainset, so excluded out
    len_classes = len(classes)

    inputs = Input(shape=(config.input_size, 1), name='input')
    layer = first_conv_block(inputs, config)
    layer = res_block_1(layer, config)
    layer = res_block_2(layer, config)
    layer = res_block_3(layer, config)
    layer = res_block_4(layer, config)

    return output_block(layer, config)
