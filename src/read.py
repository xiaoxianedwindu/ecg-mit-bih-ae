from config import get_config
import numpy as np
from utils import *
import pandas as pd

def readdata(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/train.hdf5')
    testlabelData= ddio.load('dataset/trainlabel.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    valData = ddio.load('dataset/test.hdf5')
    vallabelData= ddio.load('dataset/testlabel.hdf5')
    Xval = np.float32(valData[feature])
    yval = np.float32(vallabelData[feature])
    return (X, y, Xval, yval)

def readdata_nosplit(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/targetdata_std.hdf5')
    testlabelData= ddio.load('dataset/labeldata_std.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    return (X, y)

def main(config):
    classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']
    #print('feature:', config.feature)
    #np.random.seed(0)
    if config.split == True:
        (X,y, Xval, yval) = readdata(config.input_size, config.feature)
        print(X.shape)
        print(y.shape)
        print(pd.DataFrame(y, columns=classes).sum())
        print(Xval.shape)
        print(yval.shape)
        print(pd.DataFrame(yval, columns=classes).sum())


    else:
        (X,y) = readdata_nosplit(config.input_size, config.feature)
        print(X.shape)
        print(y.shape)
        df = pd.DataFrame(y, columns=classes)
        #print(df)
        print(df.sum())



if __name__=="__main__":
    config = get_config()
    main(config)
