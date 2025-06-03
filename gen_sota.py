from glob import glob
import scipy.io
import numpy as np
import mne
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time as t
import matplotlib.pyplot as plt
from IPython import get_ipython

import pandas as pd

import sys

import os

import matplotlib.pyplot as plt

import sota as multisnn

# Please mention the path to folder where the data set resides 

folder1= r"../Level1_4trials"    
folder2=r"../Level2_4trials"
folder3=r"../Level3_4trials"

# Results Generation

folder="../Results_Folder-3classes/"
folder= folder +"SOTA/"

if not os.path.isdir(folder):
    
    os.mkdir(folder)

    print("Creating the directory")

else:
    print("Directory already there")

easy_data = []
med_data = []
hard_data = []
precision=[]
accuracy=[]
recall=[]
f1=[]

def convertmat2mne(data):
    ch_types=['eeg']*63
    ch_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62']    
  
    info = mne.create_info(ch_names,ch_types=ch_types,sfreq=200)
    data = mne.io.RawArray(data,info)
    data.set_eeg_reference()
    epochs=mne.make_fixed_length_epochs(data,duration=3,overlap=2.4)
    return epochs.get_data().astype(np.uint8)

# remove subjects 12,20,21,7

for i in range(1,2):
    easy = folder1+f"\T{i}"
    med = folder2+f"\T{i}"
    hard = folder3+f"\T{i}"

    for temp in os.listdir(easy):

        read= os.path.join(easy,temp)
        data=scipy.io.loadmat(read)
        index=temp.replace('.mat','')

        if 'S12' in index or 'S20' in index or 'S21' in index or 'S7' in index:
            continue

        else:

            data=data[index]
            data = convertmat2mne(data)
            easy_data.append(data)

    for temp in os.listdir(med):
        
        read= os.path.join(med,temp)
        data=scipy.io.loadmat(read)
        index=temp.replace('.mat','')
        if 'S12' in index or 'S20' in index or 'S21' in index or 'S7' in index:
            continue

        else:

            data=data[index]
            data = convertmat2mne(data)
            med_data.append(data)

    for temp in os.listdir(hard):

        read= os.path.join(hard,temp)
        data=scipy.io.loadmat(read)
        index=temp.replace('.mat','')
        
        if 'S12' in index or 'S20' in index or 'S21' in index or 'S7' in index:
            continue

        else:

            data=data[index]
            data = convertmat2mne(data)
            hard_data.append(data)

easy_data_subject=easy_data
med_data_subject=med_data
hard_data_subject=hard_data

easy_epochs_labels = [len(i)*[0] for i in easy_data_subject]

med_epochs_labels = [len(i)*[1] for i in med_data_subject]
hard_epochs_labels = [len(i)*[2] for i in hard_data_subject]

data_list = easy_data_subject+med_data_subject +hard_data_subject
label_list = (easy_epochs_labels) +(med_epochs_labels) +(hard_epochs_labels)

print(len(data_list),len(label_list))

groups_list = [[i]*len(j) for i ,j in enumerate(data_list)]

# shrink data here (if need be for testing/debugging) ->
data_array = np.vstack(data_list) 
data_array = data_array.reshape((data_array.shape[0],1,data_array.shape[1],data_array.shape[2])).astype(np.uint8)
label_array = np.hstack(label_list)
group_array = np.hstack(groups_list) 

label_array=label_array.reshape(label_array.shape[0])

print(data_array.shape,label_array.shape)
print(group_array.shape)


def gen_subject(time_stamps_training,time_stamps_test):
    accuracy =multisnn.main(data_array,label_array,time_stamps_training,time_stamps_test)
   
    print(f"SOTA Accuracy: {accuracy} \n")
    
    return accuracy


vth=[]

i=0.4

vth.append(i)

time_train=[]
time_test=[]

time_train = [20, 100, 600]
time_test= [20, 100, 200, 300, 600]

param_grid= {'time_stamps_training': time_train,'time_stamps_test': time_test}

predict=[]

for train in time_train:
    for test in time_test:

        accurate = gen_subject(train,test)

        predict.append({'training steps': train, 'testing steps': test, 'accuracy':accurate })

to_save=folder+"parameters.csv"
df = pd.DataFrame(predict)
df.to_csv(to_save)
