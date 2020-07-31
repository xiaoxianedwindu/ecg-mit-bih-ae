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
            model = Model(inputs=inputs, outputs = [mu, sigma, z], name = 'Encoder')
            '''
            adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(optimizer= adam,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
            '''
            model.summary()
            return model

        inputs = Input(shape=(config.input_size, 1), name='input')
        layer = first_conv_block(inputs, config)
        return output_block(layer, config)


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
            '''
            adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(optimizer= adam,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
            '''
            model.summary()
            return model


        inputs = Input(shape=(2, ), name='decoder_input')
        layer = first_conv_block(inputs, config)
        return output_block(layer, config)