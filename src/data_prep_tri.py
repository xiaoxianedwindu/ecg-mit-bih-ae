import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from utils import *
from config import get_config

#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

config = get_config()

(X,y, group) = loaddata_nosplit_scaled_index(config.input_size, config.feature)
classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']
Xe = np.expand_dims(X, axis=2)

selected_beat_type = ['j']
#selected_subject = '207'

subject, subject_count = np.unique(group, return_counts = True)
marker = 0
beat_table = pd.DataFrame()
for c in subject_count:
    beat_table = pd.concat([beat_table, pd.DataFrame(y[marker:marker+c], columns=classes).sum()], axis=1)
    marker = marker+c
beat_table.columns = subject
beat_table = beat_table.T
print(beat_table)
beat_index_subject = np.array(beat_table.where(beat_table[selected_beat_type]>0).dropna().index)        #array of subjects with selected beat type

print(beat_index_subject)                               #subjects that have selected beat type
print(beat_table.where(beat_table[selected_beat_type]>0).sum())        #sum of beat count of subjects with selected type by beat type
print(beat_table.where(beat_table[selected_beat_type]>0).sum().sum())  #total sum of beat count of subjects with selected type
print("=========")
print(X.shape)                                          #original total beat count

X_subjects = []
y_subjects = []
X_beattype = []
y_beattype = []
X_subjects_beat = []
y_subjects_beat = []
X_trip_p = []
X_trip_s = []
X_trip_r = []

X_subjects_total = []
y_subjects_total = []
X_beattype_total = []
y_beattype_total = []
X_subjects_beat_total = []
y_subjects_beat_total = []
X_trip_p_total = []
X_trip_s_total = []
X_trip_r_total = []

for selected_beat_type in tqdm(classes):
    beat_index_subject = np.array(beat_table.where(beat_table[selected_beat_type]>0).dropna().index)        #array of subjects with selected beat type
    for selected_subject in beat_index_subject:
        top_count = 0
        for x in range(X.shape[0]):
            if group[x] == selected_subject:
                X_subjects.append(X[x])
                y_subjects.append(y[x])

            if group[x] in beat_index_subject and y[x][classes.index(selected_beat_type)] == 1:
                X_beattype.append(X[x])
                y_beattype.append(y[x])

            if group[x] == selected_subject and y[x][classes.index(selected_beat_type)] == 1:
                X_subjects_beat.append(X[x])
                y_subjects_beat.append(y[x])
                top_count+=1

            if group[x] in beat_index_subject and group[x] != selected_subject and y[x][classes.index(selected_beat_type)] == 1:
                X_trip_p.append(X[x])

            if group[x] == selected_subject and y[x][classes.index(selected_beat_type)] != 1:
                X_trip_s.append(X[x])
            
            if top_count == 500: break

                
        selected_subject_count = np.array(X_subjects).shape[0] 
        print(selected_subject_count)                    #total beat count of selected subject
        beattype_count = np.array(X_beattype).shape[0]  
        print(beattype_count)                            #total beat count of subjects with selected beat type
        subject_beat_count = np.array(X_subjects_beat).shape[0]
        print(subject_beat_count)                        #beat count of selected subject and beat
        trip_p_count = np.array(X_trip_p).shape[0]
        trip_s_count = np.array(X_trip_s).shape[0]
        print(trip_p_count, trip_s_count)
        print("=========================")
        import random
        def random_oversample(data, label, count):
            r = random.randint(0,count-1)
            data.append(data[r])
            label.append(label[r])
            return data, label

        def random_oversample_1(data, count):
            r = random.randint(0,count-1)
            data.append(data[r])
            return data

        top_count = selected_subject_count if selected_subject_count > beattype_count else beattype_count   #
        if selected_subject_count > beattype_count:
            while selected_subject_count > beattype_count:
                X_beattype, y_beattype = random_oversample(X_beattype, y_beattype, beattype_count)
                beattype_count += 1
        else:
            while beattype_count > selected_subject_count:
                X_subjects, y_subjects = random_oversample(X_subjects, y_subjects, selected_subject_count)
                selected_subject_count += 1

        while top_count > subject_beat_count:
            X_subjects_beat, y_subjects_beat = random_oversample(X_subjects_beat, y_subjects_beat, subject_beat_count)
            subject_beat_count += 1

        if trip_p_count > top_count:
            X_trip_p_temp = []
            X_trip_p_temp_count = 0
            while top_count > X_trip_p_temp_count:
                X_trip_p_temp = random_oversample_1(X_trip_p, trip_p_count)
                X_trip_p_temp_count += 1
            X_trip_p = X_trip_p_temp

        while top_count > trip_p_count:
            X_trip_p = random_oversample_1(X_trip_p, trip_p_count)
            trip_p_count += 1


        if trip_s_count > top_count:
            X_trip_s_temp = []
            X_trip_s_temp_count = 0
            while top_count > X_trip_s_temp_count:
                X_trip_s_temp = random_oversample_1(X_trip_p, trip_p_count)
                X_trip_s_temp_count += 1
            X_trip_s_temp = X_trip_s_temp

        while top_count > trip_s_count:
            X_trip_s = random_oversample_1(X_trip_s, trip_s_count)
            trip_s_count += 1

        for x in range(X.shape[0]):
            if group[x] != selected_subject and y[x][classes.index(selected_beat_type)] != 1:
                r = random.randint(0,1)
                if (r == 1): X_trip_r.append(X[x]) 
            if np.array(X_trip_r).shape[0] >= top_count:
                break
        trip_ref_count = np.array(X_trip_r).shape[0]

        print(top_count)
        def collect_values(collector, data):
            return collector.append(data)

        collect_values(X_subjects_total, X_subjects)
        collect_values(y_subjects_total, y_subjects)
        collect_values(X_beattype_total, X_beattype)
        collect_values(y_beattype_total, y_beattype)
        collect_values(X_subjects_beat_total, X_subjects_beat)
        collect_values(y_subjects_beat_total, y_subjects_beat)
        collect_values(X_trip_p_total, X_trip_p)
        collect_values(X_trip_s_total, X_trip_s)
        collect_values(X_trip_r_total, X_trip_r)


X_test = np.array([np.array(X_beattype_total), np.array(X_subjects_total), np.array(X_subjects_beat_total), np.array(X_trip_p_total), np.array(X_trip_s_total), np.array(X_trip_r_total)])
y_test = np.array([np.array(y_beattype_total), np.array(y_subjects_total), np.array(y_subjects_beat_total)])
print(X_test.shape, y_test.shape)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2], X_test.shape[3])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2], y_test.shape[3])
print(X_test.shape, y_test.shape)
print("=========================")


X_test_temp = []
for x in X_test:
  X_test_temp.append(np.expand_dims(x, axis=2))
X_test = X_test_temp

y_test_temp = []
for y in y_test:
  y_test_temp.append(np.array(pd.DataFrame(y).idxmax(axis=1)))
y_test = y_test_temp


#y = np.array(pd.DataFrame(y).idxmax(axis=1))
#y = np.column_stack((y, subject))

#from sklearn.model_selection import train_test_split
#Xe, Xvale, y, yval = train_test_split(Xe, y, test_size=0.25, random_state=1)

#(m, n) = y.shape
#y = y.reshape((m, 1, n ))
#(mvl, nvl) = yval.shape
#yval = yval.reshape((mvl, 1, nvl))

#import pandas as pd
#y = np.array(pd.DataFrame(y).idxmax(axis=1))
#yval = np.array(pd.DataFrame(yval).idxmax(axis=1))

target_train = y_test
# Reshape data
input_train = X_test

import deepdish as dd
dd.io.save('dataset/train_tri.hdf5', input_train)
dd.io.save('dataset/label_tri.hdf5', target_train)