import matplotlib.pyplot as plt
import numpy as np
import aux_functions as auxf
import os
from sklearn.model_selection import train_test_split


"""

This script processes the data from DB8 of the NinaPro dataset. https://ninapro.hevs.ch/

EMG: sliding windows and feature extraction 
Glove: sliding window and linear mapping to 5 DOA 

The transformation matrix used in aux_functions is from: 

Krasoulis, Agamemnon, Sethu Vijayakumar, and Kianoush Nazarpour. 
"Effect of user practice on prosthetic finger control with an 
intuitive myoelectric decoder." Frontiers in neuroscience 13 (2019): 461612.

https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00891/full#supplementary-material

size_val and stride_val also equal to those used in this paper

"""

user_list = [1]
dataset_list = [1, 2, 3]

size_val = 128
stride_val = 50


for user in user_list: 
    for dataset in dataset_list:

        emg = auxf.emg_process(size_val, stride_val, user, dataset)
        glove = auxf.glove_process(size_val, stride_val, user, dataset)

        # check to make sure dimensions are correct:
        if not emg.shape[0] == glove.shape[0]:
            raise ValueError("EMG and Glove data shape not aligned.")
        

# join dataset 1 and dataset 2 into trainval, then split randomly 80/20 training and validation. (they are datasets from different days)


for user in user_list: 
        

        dataset = 1

        emg_d1 = auxf.getProcessedData(user = user, dataset=dataset, mode='emg')
        glove_d1 = auxf.getProcessedData(user = user, dataset=dataset, mode='glove')

        dataset = 2

        emg_d2 = auxf.getProcessedData(user = user, dataset=dataset, mode='emg')
        glove_d2 = auxf.getProcessedData(user = user, dataset=dataset, mode='glove')

        emg_trainval = np.concatenate((emg_d1, emg_d2))
        glove_trainval = np.concatenate((glove_d1, glove_d2))

        emg_train, emg_val, glove_train, glove_val = train_test_split(emg_trainval, glove_trainval, test_size = 0.2, random_state=42)

        train_path = './datasets/train'
        auxf.ensure_directory_exists(train_path)
        val_path = './datasets/val'
        auxf.ensure_directory_exists(val_path)

        np.save(f'{train_path}/user{user}_emg_train.npy', emg_train)
        np.save(f'{val_path}/user{user}_emg_val.npy', emg_val)
        np.save(f'{train_path}/user{user}_glove_train.npy', glove_train)
        np.save(f'{val_path}/user{user}_glove_val.npy', glove_val)



