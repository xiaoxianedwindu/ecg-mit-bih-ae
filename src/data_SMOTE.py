from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from graph import ECG_model
from config import get_config
from utils import *
from read import *

from imblearn.over_sampling import SMOTE


def train(config, X, y, Xval=None, yval=None):
    
    classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']#['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S']
    if not config.split:
        from sklearn.model_selection import train_test_split
        X, Xvale, y, yval = train_test_split(X, y, test_size=0.25, random_state=12)

        print(X.shape, Xvale.shape, y.shape, yval.shape)


        sm = SMOTE(sampling_strategy = 'auto', random_state=12)
        X, y = sm.fit_sample(X, y)
        print("SMOTING")
        print(X.shape, Xvale.shape, y.shape, yval.shape)


        Xe = np.expand_dims(X, axis=2)
        (m, n) = y.shape
        y = y.reshape((m, 1, n ))
        (mvl, nvl) = yval.shape
        yval = yval.reshape((mvl, 1, nvl))
        print("RESHAPING")
        print(Xe.shape, Xvale.shape, y.shape, yval.shape)


    else:
        print(X.shape, Xval.shape, y.shape, yval.shape)
        sm = SMOTE(sampling_strategy = 'auto', random_state=12)
        X, y = sm.fit_sample(X, y)
        print("SMOTING")
        print(X.shape, Xval.shape, y.shape, yval.shape)

        Xvale = np.expand_dims(Xval, axis=2)
        Xe = np.expand_dims(X, axis=2)
        (m, n) = y.shape
        y = y.reshape((m, 1, n ))
        (mvl, nvl) = yval.shape
        yval = yval.reshape((mvl, 1, nvl))
        print("RESHAPING")
        print(Xe.shape, Xvale.shape, y.shape, yval.shape)



def main(config):
    classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']
    #print('feature:', config.feature)
    #np.random.seed(0)
    if config.split == True:
        (X,y, Xval, yval) = readdata(config.input_size, config.feature)
        print(X.shape, Xval.shape, y.shape, yval.shape)
        print(pd.DataFrame(y, columns=classes).sum())
        sm = SMOTE(sampling_strategy = 'auto', random_state=12)
        X, y = sm.fit_sample(X, y)
        print("SMOTING")
        print(X.shape, Xval.shape, y.shape, yval.shape)
        print(pd.DataFrame(y, columns=classes).sum())
        print(pd.DataFrame(yval, columns=classes).sum())


    else:
        (X,y) = readdata_nosplit(config.input_size, config.feature)

        from sklearn.model_selection import train_test_split
        X, Xvale, y, yval = train_test_split(X, y, test_size=0.25, random_state=12)

        print(X.shape, Xvale.shape, y.shape, yval.shape)
        sm = SMOTE(sampling_strategy = 'auto', random_state=12)
        X, y = sm.fit_sample(X, y)
        print("SMOTING")
        print(X.shape, Xvale.shape, y.shape, yval.shape)
        print(pd.DataFrame(y, columns=classes).sum())


if __name__=="__main__":
    config = get_config()
    main(config)
