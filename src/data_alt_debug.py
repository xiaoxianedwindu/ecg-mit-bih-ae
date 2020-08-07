"""
The data is provided by 
https://physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm

The recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range.
Two or more cardiologists independently annotated each record; disagreements were resolved to obtain the computer-readable
reference annotations for each beat (approximately 110,000 annotations in all) included with the database.

    Code		Description
    N		Normal beat (displayed as . by the PhysioBank ATM, LightWAVE, pschart, and psfd)
    L		Left bundle branch block beat
    R		Right bundle branch block beat
    B		Bundle branch block beat (unspecified)
    A		Atrial premature beat
    a		Aberrated atrial premature beat
    J		Nodal (junctional) premature beat
    S		Supraventricular premature or ectopic beat (atrial or nodal)
    V		Premature ventricular contraction
    r		R-on-T premature ventricular contraction
    F		Fusion of ventricular and normal beat
    e		Atrial escape beat
    j		Nodal (junctional) escape beat
    n		Supraventricular escape beat (atrial or nodal)
    E		Ventricular escape beat
    /		Paced beat
    f		Fusion of paced and normal beat
    Q		Unclassifiable beat
    ?		Beat not classified during learning
"""

from __future__ import division, print_function
import os
from tqdm import tqdm
import numpy as np
import random
from utils import *
from config import get_config

def preprocess( split ):
    nums = ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
    features = ['MLII', 'V1', 'V2', 'V4', 'V5'] 

    if split :
        testset = ['104', '113', '119', '208', '210']
        trainset = [x for x in nums if x not in testset]

    def dataSaver(dataSet, datasetname, labelsname):
        classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']#['N','V','/','A','F','~']#,'L','R',f','j','E','a']#,'J','Q','e','S']  #['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']
        Nclass = len(classes)
        datadict, datalabel= dict(), dict()
        index = []

        for feature in features:
            datadict[feature] = list()
            datalabel[feature] = list()

        def dataprocess():
            input_size = config.input_size
            #test = ['100']
            test = ['100','101', '103']
            for num in tqdm(test):
            #num= '101'
                from wfdb import rdrecord, rdann, rdsamp
                record = rdrecord('dataset/'+ num, smooth_frames= True)
                r = rdsamp('dataset/'+ num)
                #print(r)

                
                signal0 = np.nan_to_num(np.array(r[0][:,0]))
                signal1 = np.nan_to_num(np.array(r[0][:,1]))
                
                '''
                print('signal')
                print(signal0.mean(), signal1.mean())
                print(signal0.std(), signal1.std())
                print(min(signal0), max(signal0)) 
                print(min(signal1), max(signal1))
                '''

                signal0= ((signal0 - signal0.mean())/ signal0.std())
                signal1= ((signal1 - signal1.mean())/ signal1.std())
                '''
                print('subtract mean and divide by std')
                print(signal0.mean(), signal1.mean())
                print(signal0.std(), signal1.std())
                print(min(signal0), max(signal0)) 
                print(min(signal1), max(signal1))
                print("=======================")
                '''

                signal0 = signal0.tolist()
                signal1 = signal1.tolist()

                #from sklearn.preprocessing import minmax_scale
                #signal0 = minmax_scale(np.nan_to_num(np.array(r[0][:,0]))).tolist()
                #signal1 = minmax_scale(np.nan_to_num(np.array(r[0][:,1]))).tolist()

                '''
                from sklearn.preprocessing import normalize
                signal0 = np.nan_to_num(np.array(r[0][:,0]))
                print(signal0.shape)
                signal0 = normalize(np.nan_to_num(np.array(r[0][:,0])).reshape(-1,1)).tolist()
                signal1 = normalize(np.nan_to_num(np.array(r[0][:,1])).reshape(-1,1)).tolist()
                print(len(signal0))
            
                signal0 = np.nan_to_num(np.array(r[0][:,0]))
                signal1 = np.nan_to_num(np.array(r[0][:,1]))
                signal0 = (signal0 - min(signal0) / max(signal0) - min(signal0)) 
                signal1 = (signal1 - min(signal1) / (max(signal1) - min(signal1)))
                signal0 = signal0.tolist()
                signal1 = signal1.tolist()
                '''


                #print('signal')
                #print(min(signal0), max(signal0)) 
                #print(min(signal1), max(signal1))

                ann = rdann('dataset/'+ num, extension='atr')

                r_peaks = ann.sample[0:-1]
                #labels = ann.symbol[1:-1]
                
                #print('r_peaks')
                #print(r_peaks)
                #print('labels')
                #print(len(labels))
                global counter
                counter = 0
                feature0, feature1 = record.sig_name[0], record.sig_name[1]

                global lppened0, lappend1, dappend0, dappend1 
                lappend0 = datalabel[feature0].append
                lappend1 = datalabel[feature1].append
                dappend0 = datadict[feature0].append
                dappend1 = datadict[feature1].append
                # skip a first peak to have enough range of the sample 
                for peak in r_peaks[1:-1]:
                #for peak in tqdm(r_peaks[1:-1]):
                    start, end =  peak-input_size//2 , peak+input_size//2
                    if start < 0: start = 0
                    peak_actual = np.array(signal0[start:end]).argmax() - input_size//2 + peak
                    start, end = peak_actual-input_size//2 , peak_actual+input_size//2
                    if start < 0: start = 0
                    
                    #print(np.array(signal0[start:end]).max())
                    #print(np.array(signal0[start:end]).argmax())

                    ann = rdann('dataset/'+ num, extension='atr', sampfrom = start, sampto = end, return_label_elements=['symbol'])
                    
                    def to_dict(chosenSym):
                        y = [0]*Nclass
                        y[classes.index(chosenSym)] = 1
                        lappend0(y)
                        lappend1(y)
                        dappend0(signal0[start:end])
                        dappend1(signal1[start:end])

                    annSymbol = ann.symbol
                    if len(annSymbol) == 1:
                        #print(annSymbol[0])
                        #print('swapping symbol')
                        if annSymbol[0] == "/":
                            #print("'/' detected")
                            annSymbol[0] = 'P'
                        #print(annSymbol[0])

                    # remove some of "N" which breaks the balance of dataset 
                    #if len(annSymbol) == 1 and (annSymbol[0] in classes) and (annSymbol[0] != "N" or np.random.random()<0.15):
                    if len(annSymbol) == 1 and (annSymbol[0] in classes):
                        to_dict(annSymbol[0])
                        counter += 1

                    
                index.append((num, counter))

        
        dataprocess()
        #noises = add_noise(config)
        for feature in ['MLII', 'V1', 'V2', 'V4', 'V5']: 
            datadict[feature]=np.array(datadict[feature])
            datalabel[feature] = np.array(datalabel[feature])
        print(index)


        import deepdish as dd
        dd.io.save(datasetname, datadict)
        dd.io.save(labelsname, datalabel)
        dd.io.save('dataset/index_debug.hdf5', index)

    if split:
        dataSaver(trainset, 'dataset/train_debug.hdf5', 'dataset/trainlabel_debug.hdf5')
        dataSaver(testset, 'dataset/test_debug.hdf5', 'dataset/testlabel_debug.hdf5')
    else:
        dataSaver(nums, 'dataset/targetdata_debug.hdf5', 'dataset/labeldata_debug.hdf5')

def main(config):
    def Downloadmitdb():
        ext = ['dat', 'hea', 'atr']
        nums = ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
        for num in tqdm(nums):
            for e in ext:
                url = "https://physionet.org/physiobank/database/mitdb/"
                url = url + num +"."+e
                mkdir_recursive('dataset')
                cmd = "cd dataset && curl -O "+url
                os.system(cmd)

    if config.downloading:
        Downloadmitdb()
    return preprocess(config.split)

if __name__=="__main__":
    config = get_config()
    main(config)
