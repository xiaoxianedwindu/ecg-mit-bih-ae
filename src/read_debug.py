from config import get_config
import numpy as np
from utils import *
import pandas as pd

def readdata(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/train_debug.hdf5')
    testlabelData= ddio.load('dataset/trainlabel_debug.hdf5')
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
    #trainData = ddio.load('dataset/targetdata_std.hdf5')
    #testlabelData= ddio.load('dataset/labeldata_std.hdf5')
    #indexData= ddio.load('dataset/index_std.hdf5')
    trainData = ddio.load('dataset/targetdata_debug.hdf5')
    testlabelData= ddio.load('dataset/labeldata_debug.hdf5')
    indexData= ddio.load('dataset/index_debug.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    subjectLabel = (np.array(pd.DataFrame(indexData)[1]))
    group = []
    for x in subjectLabel:
        for beat in range(x):    
            group.append(x)
    group = np.array(group)
    print(np.unique(group, return_counts = True))
    return (X, y, group)

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
        #(X,y, groups) = readdata_nosplit(128, config.feature)
        (X,y, groups) = readdata_nosplit(config.input_size, config.feature)
        Xe = np.expand_dims(X, axis=2)
        #(m, n) = y.shape
        #y = y.reshape((m, 1, n ))
        print(X.shape)
        print(y.shape)
        df = pd.DataFrame(y, columns=classes)
        #print(df)
        print(df.sum())
        y = df.idxmax(axis=1)
        print (pd.unique(y))


        from matplotlib import pyplot as plt
        print(X.min(), X.max())
        xaxis = np.arange(0,config.input_size)
        for count in range(5):
            plt.plot(xaxis, X[2265+count])
        plt.show()
        
        '''
        from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
        logo = LeaveOneGroupOut()
        for train, test in logo.split(X, y, groups=groups):
            #print("%s %s" % (train, test))
            print("========")
            print(train.shape)
            print(test.shape)

        from keras.models import Model, Sequential
        from keras.layers import Input, Conv1D, Dense, add, Flatten, Dropout,MaxPooling1D, Activation, BatchNormalization, Lambda
        from keras.optimizers import Adam
        def first_conv_block(inputs, config):
            layer = Conv1D(filters=config.filter_length,
                kernel_size=config.kernel_size,
                padding='same',
                strides=1,
                kernel_initializer='he_normal')(inputs)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)

            shortcut = MaxPooling1D(pool_size=1,
                        strides=1)(layer)

            layer =  Conv1D(filters=config.filter_length,
                kernel_size=config.kernel_size,
                padding='same',
                strides=1,
                kernel_initializer='he_normal')(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(config.drop_rate)(layer)
            layer =  Conv1D(filters=config.filter_length,
                            kernel_size=config.kernel_size,
                            padding='same',
                            strides=1,
                            kernel_initializer='he_normal')(layer)
            return add([shortcut, layer])
        def main_loop_blocks(layer, config):
            filter_length = config.filter_length
            def zeropad(x):
                """ 
                zeropad and zeropad_output_shapes are from 
                https://github.com/awni/ecg/blob/master/ecg/network.py
                """
                y = K.zeros_like(x)
                return K.concatenate([x, y], axis=2)

            def zeropad_output_shape(input_shape):
                shape = list(input_shape)
                assert len(shape) == 3
                shape[2] *= 2
                return tuple(shape)

            block_index = 1
            subsample_length = 2 if block_index % 2 == 0 else 1
            shortcut = MaxPooling1D(pool_size=subsample_length)(layer)

            # 5 is chosen instead of 4 from the original model
            if block_index % 4 == 0 and block_index > 0 :
                # double size of the network and match the shapes of both branches
                shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
                filter_length *= 2

            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer =  Conv1D(filters= filter_length,
                            kernel_size=config.kernel_size,
                            padding='same',
                            strides=subsample_length,
                            kernel_initializer='he_normal')(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(config.drop_rate)(layer)
            layer =  Conv1D(filters= filter_length,
                            kernel_size=config.kernel_size,
                            padding='same',
                            strides= 1,
                            kernel_initializer='he_normal')(layer)
            layer = add([shortcut, layer])
            return layer

        def create_model():
            classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']
            len_classes = len(classes)

            inputs = Input(shape=(config.input_size, 1), name='input')
            layer = first_conv_block(inputs, config)
            layer = main_loop_blocks(layer, config)
            from keras.layers.wrappers import TimeDistributed
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            #layer = Flatten()(layer)
            outputs = TimeDistributed(Dense(len_classes, activation='softmax'))(layer)
            model = Model(inputs=inputs, outputs=outputs)
            
            adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(optimizer= adam,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
            model.summary()
            return model

        from keras.wrappers.scikit_learn import KerasClassifier
        #model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=config.batch)
        #clf = model
        cv = logo.split(X, y, groups=groups)
        from sklearn import svm
        clf = svm.SVC(kernel='linear', C=1)
        scores = cross_val_score(clf, X, y,groups = groups, cv=cv)
        print(scores)
        '''



if __name__=="__main__":
    config = get_config()
    main(config)
