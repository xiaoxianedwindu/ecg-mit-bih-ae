from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from graph import ECG_model
from graph_ae import encoder_model, decoder_model, vae_model
from config import get_config
from utils import *
import pandas as pd
from keras.layers import Input, LSTM, RepeatVector, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam




def train(config, X, y, Xval=None, yval=None):
    
    classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']#['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S']
    #Xe = np.expand_dims(X, axis=2)
    if not config.split:
        from sklearn.model_selection import train_test_split
        #Xe, Xvale, y, yval = train_test_split(Xe, y, test_size=0.25, random_state=1)
        X, Xval, y, yval = train_test_split(X, y, test_size=0.25, random_state=1)
        
        #(m, n) = y.shape
        #y = y.reshape((m, 1, n ))
        #(mvl, nvl) = yval.shape
        #yval = yval.reshape((mvl, 1, nvl))
        
    else:
        Xvale = np.expand_dims(Xval, axis=2)
        (m, n) = y.shape
        y = y.reshape((m, 1, n ))
        (mvl, nvl) = yval.shape
        yval = yval.reshape((mvl, 1, nvl))

    if config.checkpoint_path is not None:
        model = model.load_model(config.checkpoint_path)
        initial_epoch = config.resume_epoch # put the resuming epoch
    else:
        #encoder = encoder_model(config)
        #decoder = decoder_model(config)
        initial_epoch = 0

    mkdir_recursive('models')
    #lr_decay_callback = LearningRateSchedulerPerBatch(lambda epoch: 0.1)
    callbacks = [
            EarlyStopping(patience = config.patience, verbose=1),
            ReduceLROnPlateau(factor = 0.5, patience = 3, min_lr = 0.01, verbose=1),
            TensorBoard( log_dir='./logs', histogram_freq=0, write_graph = True, write_grads=False, write_images=True),
            ModelCheckpoint('models/{}-latest.hdf5'.format(config.feature), monitor='val_loss', save_best_only=False, verbose=1, period=10)
            # , lr_decay_callback
    ]
    print('=====shapes======')
    print("Xe shape", X.shape)
    print("y shape", y.shape)
    print('=================')
    

    timesteps = config.input_size
    inputs = Input(shape=(timesteps, 1))    
    encoded = LSTM(timesteps, activation=LeakyReLU(alpha=0.2))(inputs)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(timesteps, activation=LeakyReLU(alpha=0.2))(decoded)

    autoencoder = Model(inputs, decoded)
    #autoencoder = Model(inputs, encoded)

    adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    autoencoder.compile(optimizer= adam,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    autoencoder.summary()

    autoencoder.fit(X, X,
            validation_data=(Xval, Xval),
            epochs=3,
            batch_size=config.batch,
            callbacks=callbacks,
            initial_epoch=initial_epoch)

    Xae = autoencoder.predict(X)
    Xvalae = autoencoder.predict(Xval)

    #print_results(config, decoder, Xvalee, yval, classes, )
    model = ECG_model(config)
    model.fit(Xae, y,
            validation_data=(Xvalae, yval),
            epochs=config.epochs,
            batch_size=config.batch,
            callbacks=callbacks,
            initial_epoch=initial_epoch)


    from matplotlib import pyplot as plt
    xaxis = np.arange(1,257)
    plt.plot(xaxis, Xval[0,...])
    plt.plot(xaxis, Xvalae[0,...])
    plt.show()


    print_results(config, model, Xvalae, yval, classes, )

    '''
    vae = vae_model(config)

    vae.fit(Xe, Xe,
            validation_data=(Xvale, Xvale),
            epochs=config.epochs,
            batch_size=config.batch,
            callbacks=callbacks,
            initial_epoch=initial_epoch)   
    '''



    '''
    model.fit(Xe, y,
            validation_data=(Xvale, yval),
            epochs=config.epochs,
            batch_size=config.batch,
            callbacks=callbacks,
            initial_epoch=initial_epoch)
    print_results(config, model, Xvale, yval, classes, )
    '''
    #return model


def main(config):
    print('feature:', config.feature)
    #np.random.seed(0)
    if config.split == True:
        (X,y, Xval, yval) = loaddata(config.input_size, config.feature)
        train(config, X, y, Xval, yval)

    else:
        (X,y) = loaddata_nosplit(config.input_size, config.feature)
        train(config, X, y)


if __name__=="__main__":
    config = get_config()
    main(config)


