import matplotlib.pyplot as plt
import numpy as np
import aux_functions as auxf
import os


"""

This script processes the data from DB8 of the NinaPro dataset. https://ninapro.hevs.ch/

EMG: sliding windows and feature extraction 
Glove: sliding window and linear mapping to 5 DOA 
Restimulus data: sliding windows


The transformation matrix is from: 

Krasoulis, Agamemnon, Sethu Vijayakumar, and Kianoush Nazarpour. 
"Effect of user practice on prosthetic finger control with an 
intuitive myoelectric decoder." Frontiers in neuroscience 13 (2019): 461612.

https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00891/full#supplementary-material

"""



## ----- pre-processing constants ------ ## 


# winsize = 128, stride length = 52ms (about 40% overlap)

fs = 2000 # 
win_size = 120
win_stride = 50
cutoff_f = 450
order = 4


list_users = [1]
feat_ID_list = ['all']

dataset_list = [1, 2, 3]

## ----- functions ------ ## 



def runEMG():
    for user in list_users:
        for dataset in dataset_list:

            auxf.emg_process(cutoff_val = cutoff_f, size_val = win_size, stride_val = win_stride, user = user, dataset=dataset, order = order)


def runLabels():
    for user in list_users:
        for dataset in dataset_list:

            rawBool = False
            auxf.glove_process(size_val = win_size, stride_val = win_stride, user = user, dataset = dataset, rawBool = rawBool)
            auxf.restimulusProcess(size_val= win_size, stride_val= win_stride, user = user, dataset= dataset, rawBool = rawBool)


# ---------- run the processing ------------#

runEMG()
runLabels()


directory = './processed_data'


for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        data = np.load(file_path)
        print('data shape', data.shape)



# ----------- perform linear mapping for the glove data --------------- #



A_T = np.array([
    [0.639, 0, 0, 0, 0], 
    [0.383, 0, 0, 0, 0], 
    [0, 1, 0, 0, 0], 
    [-0.639, 0, 0, 0, 0], 
    [0, 0, 0.4, 0 , 0], 
    [0, 0, 0.6, 0, 0], 
    [0, 0, 0, 0.4, 0], 
    [0, 0, 0, 0.6, 0], 
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.1667], 
    [0, 0, 0, 0, 0.3333], 
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.1667], 
    [0, 0, 0, 0, 0.3333], 
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0], 
    [-0.19, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
])

# glove = auxf.getProcessedData(user = 1, dataset = 1, mode = 'glove', rawBool = False)

# A = A_T.T

# glove_mapped = A@glove.T

# user_list = np.arange(1, 13)
# dataset_list = [1, 2, 3]

# for user in user_list:
#     for dataset in dataset_list:
#         glove_data = auxf.getProcessedData(user = user, dataset = dataset, mode = 'glove', rawBool=False)

#         glove_mapped = A @ glove_data.T

#         directory = f"./processed_data/glove_mapped/"

#         auxf.ensure_directory_exists(directory)
#         np.save(f"{directory}/user{user}_dataset{dataset}.npy", glove_mapped)

