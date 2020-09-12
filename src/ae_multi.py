'''
  Variational Autoencoder (VAE) with the Keras Functional API.
'''

import keras
from keras.layers import Conv1D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, UpSampling1D, AveragePooling1D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy, mean_squared_error
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from config import get_config

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same', activation='relu', name='conv12d'):
    """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding, activation='relu', name = name)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x


# Load MNIST dataset
#(input_train, target_train), (input_test, target_test) = mnist.load_data()

config = get_config()

(X,y) = loaddata_nosplit_scaled(config.input_size, config.feature)
classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']#['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S']
Xe = np.expand_dims(X, axis=2)
from sklearn.model_selection import train_test_split
Xe, Xvale, y, yval = train_test_split(Xe, y, test_size=0.25, random_state=1)
#(m, n) = y.shape
#y = y.reshape((m, 1, n ))
#(mvl, nvl) = yval.shape
#yval = yval.reshape((mvl, 1, nvl))
import pandas as pd
y = np.array(pd.DataFrame(y).idxmax(axis=1))
yval = np.array(pd.DataFrame(yval).idxmax(axis=1))

target_train = y
target_test = yval 
# Data & model configuration
batch_size = 256
no_epochs = 50
validation_split = 0.2
verbosity = 1
latent_dim = 2
num_channels = 1

# Reshape data

input_train = Xe
input_test = Xvale
input_shape = (config.input_size, 1)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')


# # =================
# # Encoder
# # =================

# Definition
i       = Input(shape=input_shape, name='encoder_input')
cx      = Conv1D(filters=8, kernel_size=16, strides=2, padding='same', activation='relu')(i)
cx      = BatchNormalization()(cx)
cx      = Conv1D(filters=16, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
cx      = BatchNormalization()(cx)
cx      = Conv1D(filters=16, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
cx      = BatchNormalization()(cx)
cx      = Conv1D(filters=1, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
#cx      = BatchNormalization()(cx)
#cx      = Conv1D(filters=1, kernel_size=8, strides=2, padding='same', activation='relu')(cx)
eo      = BatchNormalization()(cx)

#x       = Flatten()(cx)
#x       = Dense(20, activation='relu')(x)
#x       = BatchNormalization()(x)
#mu      = Dense(latent_dim, name='latent_mu')(x)
#sigma   = Dense(latent_dim, name='latent_sigma')(x)

# Get Conv2D shape for Conv2DTranspose operation in decoder
conv_shape = K.int_shape(cx)
print(conv_shape)
# Define sampling with reparameterization trick
def sample_z(args):
  mu, sigma = args
  batch     = K.shape(mu)[0]
  dim       = K.int_shape(mu)[1]
  eps       = K.random_normal(shape=(batch, dim))
  return mu + K.exp(sigma / 2) * eps

# Use reparameterization trick to ....??
#z       = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([mu, sigma])

# Instantiate encoder
encoder = Model(i, eo, name='encoder')
#encoder = Model(i, [mu, sigma, z], name='encoder')
encoder.summary()

# # =================
# # Encoder_2
# # =================

# Definition
i_2       = Input(shape=input_shape, name='encoder2_input')
cx      = Conv1D(filters=8, kernel_size=16, strides=2, padding='same', activation='relu')(i_2)
cx      = BatchNormalization()(cx)
cx      = Conv1D(filters=16, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
cx      = BatchNormalization()(cx)
cx      = Conv1D(filters=16, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
cx      = BatchNormalization()(cx)
cx      = Conv1D(filters=1, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
#cx      = BatchNormalization()(cx)
#cx      = Conv1D(filters=1, kernel_size=8, strides=2, padding='same', activation='relu')(cx)
eo_2      = BatchNormalization()(cx)

#x       = Flatten()(cx)
#x       = Dense(20, activation='relu')(x)
#x       = BatchNormalization()(x)
#mu      = Dense(latent_dim, name='latent_mu')(x)
#sigma   = Dense(latent_dim, name='latent_sigma')(x)

# Get Conv2D shape for Conv2DTranspose operation in decoder
conv_shape_2 = K.int_shape(cx)

# Instantiate encoder
encoder_2 = Model(i_2, eo_2, name='encoder_2')
encoder_2.summary()

# =================
# Decoder
# =================

# Definition
#d_i   = Input(shape=(latent_dim, ), name='decoder_input')
#d_i   = Input(shape=(conv_shape[1]*2, conv_shape[2]), name='decoder_input')
d_i   = Input(shape=(conv_shape[1], conv_shape[2]*2), name='decoder_input')
#x     = Dense(conv_shape[1] * conv_shape[2], activation='relu')(d_i)
#x     = BatchNormalization()(x)
#x     = Reshape((conv_shape[1], conv_shape[2]))(x)
#x     = Reshape((conv_shape[1], conv_shape[2]))(d_i)
cx    = UpSampling1D(size=2)(d_i)
#cx    = UpSampling1D(size=2)(cx)
cx    = Conv1D(filters=2, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
cx    = BatchNormalization()(cx)
cx    = UpSampling1D(size=2)(cx)
cx    = Conv1D(filters=2, kernel_size=16, strides=2, padding='same', activation='relu')(cx)
cx    = BatchNormalization()(cx)
cx    = UpSampling1D(size=2)(cx)
'''
cx    = Conv1D(filters=1, kernel_size=16, strides=2, padding='same',  activation='relu', name = 'conv12d2')(cx)
cx    = BatchNormalization()(cx)
cx    = UpSampling1D(size=2)(cx)
'''
cx    = Conv1D(filters=1, kernel_size=16, strides=2, padding='same',  activation='relu', name = 'conv12d3')(cx)
cx    = BatchNormalization()(cx)
cx    = UpSampling1D(size=4)(cx)
cx    = Conv1D(filters=1, kernel_size=16, strides=2, padding='same',  activation='relu', name = 'conv12d4')(cx)
cx    = BatchNormalization()(cx)
cx    = UpSampling1D(size=4)(cx)
cx    = Conv1D(filters=num_channels, kernel_size=16, activation='relu', padding='same', name='decoder_output')(cx)
o     = UpSampling1D(size=2)(cx)


'''
cx    = Conv1DTranspose(d_i, filters=16, kernel_size=3, strides=2, padding='same', activation='relu')#(x)
cx    = BatchNormalization()(cx)
cx    = Conv1DTranspose(cx, filters=8, kernel_size=3, strides=2, padding='same',  activation='relu', name = 'conv12d2')#(cx)
cx    = BatchNormalization()(cx)
cx    = Conv1DTranspose(cx, filters=8, kernel_size=3, strides=2, padding='same',  activation='relu', name = 'conv12d3')#(cx)
cx    = BatchNormalization()(cx)
cx    = Conv1DTranspose(cx, filters=8, kernel_size=3, strides=2, padding='same',  activation='relu', name = 'conv12d4')#(cx)
cx    = BatchNormalization()(cx)
o     = Conv1DTranspose(cx, filters=num_channels, kernel_size=3, activation='relu', padding='same', name='decoder_output')#(cx)
'''
# Instantiate decoder
decoder = Model(d_i, o, name='decoder')
decoder.summary()

# =================
# VAE as a whole
# =================
from tensorflow import concat
# Instantiate VAE
vae_outputs = decoder(concat([encoder(i), encoder_2(i)], 2))
#vae_outputs = decoder(encoder(i))
#vae_outputs = decoder(encoder(i)[2])
vae         = Model(i, vae_outputs, name='multi-ae')
vae.summary()

# Define loss
def kl_reconstruction_loss(true, pred):
  # Reconstruction loss
  reconstruction_loss = mean_squared_error(K.flatten(true), K.flatten(pred))# * 256
  # reconstruction_loss = K.flatten(true)-K.flatten(pred)# * 256
  # KL divergence loss
  ''' 
  kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  # Total loss = 50% rec + 50% KL divergence loss
  '''
  return reconstruction_loss
  #return K.mean(reconstruction_loss + kl_loss)

# Compile VAE
vae.compile(optimizer='adam', loss=kl_reconstruction_loss)

# Train autoencoder
vae.fit(input_train, input_train, epochs = no_epochs, batch_size = batch_size, validation_data = (input_test, input_test))
#vae.fit(input_train, input_train, epochs = no_epochs, batch_size = batch_size, validation_split = validation_split)


# =================
# Results visualization
# Credits for original visualization code: https://keras.io/examples/variational_autoencoder_deconv/
# (François Chollet).
# Adapted to accomodate this VAE.
# =================
def viz_latent_space(encoder, data):
  input_data, target_data = data

  print(target_data.shape)
  #print(target_data.shape[0])
  #print(target_data)
  print(encoder.predict(input_data).shape)#.reshape((32,16))
  print(encoder.predict(input_data).reshape(input_data.shape[0], 16).shape)#.reshape((32,16))

  from sklearn.manifold import TSNE
  X_tsne = TSNE(n_components=2, random_state=1).fit_transform(encoder.predict(input_data).reshape(input_data.shape[0], 16))
  print(X_tsne.shape)
  print(X_tsne)


  plt.figure(figsize=(8, 10))
  scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=target_data, label = classes)
  plt.legend(handles=scatter.legend_elements()[0], labels=classes)
  plt.title("tsne")
  plt.show()

def viz_latent_space_pca(encoder, data):
  input_data, target_data = data

  print(target_data.shape)
  #print(target_data.shape[0])
  #print(target_data)
  print(encoder.predict(input_data).shape)#.reshape((32,16))
  print(encoder.predict(input_data).reshape(input_data.shape[0], 16).shape)#.reshape((32,16))

  from sklearn.decomposition import PCA
  principalComponents = PCA(n_components=2, random_state = 1).fit_transform(encoder.predict(input_data).reshape(input_data.shape[0], 16)) 
  print(principalComponents.shape)
  print(principalComponents)   


  plt.figure(figsize=(8, 10))
  scatter = plt.scatter(principalComponents[:,0], principalComponents[:,1], c=target_data, label=classes)
  plt.legend(handles=scatter.legend_elements()[0], labels=classes)
  plt.title("pca")
  plt.show()


def viz_decoded(encoder, decoder, data):
  num_samples = 15
  figure = np.zeros((img_width * num_samples, img_height * num_samples, num_channels))
  grid_x = np.linspace(-4, 4, num_samples)
  grid_y = np.linspace(-4, 4, num_samples)[::-1]
  for i, yi in enumerate(grid_y):
      for j, xi in enumerate(grid_x):
          z_sample = np.array([[xi, yi]])
          x_decoded = decoder.predict(z_sample)
          digit = x_decoded[0].reshape(img_width, img_height, num_channels)
          figure[i * img_width: (i + 1) * img_width,
                  j * img_height: (j + 1) * img_height] = digit
  plt.figure(figsize=(10, 10))
  start_range = img_width // 2
  end_range = num_samples * img_width + start_range + 1
  pixel_range = np.arange(start_range, end_range, img_width)
  sample_range_x = np.round(grid_x, 1)
  sample_range_y = np.round(grid_y, 1)
  plt.xticks(pixel_range, sample_range_x)
  plt.yticks(pixel_range, sample_range_y)
  plt.xlabel('z - dim 1')
  plt.ylabel('z - dim 2')
  # matplotlib.pyplot.imshow() needs a 2D array, or a 3D array with the third dimension being of shape 3 or 4!
  # So reshape if necessary
  fig_shape = np.shape(figure)
  if fig_shape[2] == 1:
    figure = figure.reshape((fig_shape[0], fig_shape[1]))
  # Show image
  plt.imshow(figure)
  plt.show()

def plot_some_signals(vae, data):
    x_vae_pred = vae.predict(data)

    from matplotlib import pyplot as plt
    xaxis = np.arange(0,config.input_size)
    for count in range(5):
        plt.plot(xaxis, x_vae_pred[count])
    plt.title("ae reconstructed beats")
    plt.xlabel("beat length")
    plt.ylabel("signal")
    plt.show()

# Plot results
data = (input_test, target_test)
viz_latent_space(encoder, data)
viz_latent_space(encoder_2, data)
viz_latent_space_pca(encoder, data)
viz_latent_space_pca(encoder2, data)

plot_some_signals()
#viz_decoded(encoder, decoder, data)