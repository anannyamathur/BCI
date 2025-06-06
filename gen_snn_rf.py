import scipy.io
import numpy as np
import mne

import pandas as pd
import os
import rf_snn as multisnn

import argparse
import yaml

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

parser = argparse.ArgumentParser(description="Running neural arch with config")
parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config YAML file')

args = parser.parse_args()
config = load_config(args.config)

classes= int(config['dataset']['classes'])
multisnn.classes= classes

# load folder names 
folders= dict()

for i in range(1, classes+1):
    folders[f"class{i}"]= config['dataset'][f"class{i}"]

# Results Generation

folder= config["results"]["path"]

if not os.path.isdir(folder):
    
    os.mkdir(folder)

    print("Creating the directory")

else:
    print("Directory already there")

def convertmat2mne(data):
    ch_types=['eeg']*63
    ch_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62']    
  
    info = mne.create_info(ch_names,ch_types=ch_types,sfreq=200)
    data = mne.io.RawArray(data,info)
    data.set_eeg_reference()
    epochs=mne.make_fixed_length_epochs(data,duration=3,overlap=2.4)
    return epochs.get_data().astype(np.uint8)

# remove subjects 12,20,21,7

data_list=[]
label_list=[]

class_id=0

for id in folders:

    level= folders[id] + f"/T{1}"
    eeg_data_level= []

    for temp in os.listdir(level):

        read= os.path.join(level,temp)
        data=scipy.io.loadmat(read)
        index=temp.replace('.mat','')

        if 'S12' in index or 'S20' in index or 'S21' in index or 'S7' in index:
            continue

        else:

            data=data[index]
            data = convertmat2mne(data)
            eeg_data_level.append(data)
        
    data_list= data_list + eeg_data_level

    labels = [len(i)*[class_id] for i in eeg_data_level]
    label_list= label_list + labels
    
    class_id = class_id +1 
 
print(len(data_list),len(label_list))

# shrink data here (if need be for testing/debugging) ->
data_array = np.vstack(data_list) 
data_array = data_array.reshape((data_array.shape[0],1,data_array.shape[1],data_array.shape[2])).astype(np.uint8)
label_array = np.hstack(label_list)

label_array=label_array.reshape(label_array.shape[0])

print(data_array.shape,label_array.shape)

bias=0

layers= config["snn"]["layers"]
hidden_layers= [int(x) for x in layers]

def gen_subject(hidden_layers,time_stamps_training,time_stamps_test,learning,decay,vth,pre,post,posttau,pretau,bias):

    threshold=vth

    accuracy, accuracy_rf, accuracy_snn=multisnn.main(data_array,label_array,hidden_layers,time_stamps_training,time_stamps_test,learning,decay,threshold,pre,post,posttau,pretau,bias)
    
    print(f"RF Accuracy: {accuracy_rf} \n")
    
    print(f"SNN-RF Accuracy: {accuracy} \n")

    print(f"SNN Accuracy: {accuracy_snn} \n")
    
    return accuracy, accuracy_rf, accuracy_snn

vth=[]

i=0.4

vth.append(i)

time_train = config["training"]["steps"]
time_test= config["testing"]["steps"]

param_grid= {'vth': vth, 'time_stamps_training': time_train,'time_stamps_test': time_test}

learning= 0.01
decay= 0.1

pre=0.01
post=0.1
posttau=2
pretau=1

predict=[]

for threshold in vth:
    for train in time_train:
        for test in time_test:

            accurate, accurate_rf, accurate_snn= gen_subject(hidden_layers,train,test,learning,decay,threshold,pre,post,posttau,pretau,bias)

            predict.append({'vth': threshold, 'training steps': train, 'testing steps': test, 'accuracy_rf':accurate_rf,'accuracy': accurate,'accuracy_snn': accurate_snn})

to_save=folder+"parameters.csv"
df = pd.DataFrame(predict)
df.to_csv(to_save)
