import matplotlib.pyplot as plt
import numpy as np
import aux_functions as auxf
import os


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