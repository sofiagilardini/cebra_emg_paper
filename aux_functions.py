
import matplotlib.pyplot as plt
import scipy.io 
import numpy as np 
# import cebra
# from cebra import CEBRA
import cebra.models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.signal import butter, lfilter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import os
from sklearn.preprocessing import StandardScaler
from scipy import stats 

def getdata(user, dataset):
    data = scipy.io.loadmat(f"./datasets/S{user}_E1_A{dataset}.mat")
    for key in data:
        print(key)
    emg_data = data['emg']
    glove_data = data['glove']
    restimulus_data = data['restimulus']

    return emg_data, glove_data, restimulus_data



# ----------- FILTERING ------------------------------#

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# --------- FILTERING END -------------------------#

def getProcessedData(user: int, dataset: int, mode: str):

    """
    mode = ["glove", "emg"]
    
    """

    data = np.load(f"./processed_data/{mode}/dataset{dataset}/{mode}_{user}_{dataset}.npy")

    return data


## --------- SLIDING WINDOW FUNCTIONS ---------------- #

def slidingWindowEMG(x, win_size, win_stride):

    # Perform the sliding window operation
    num_windows = (len(x) - win_size) // win_stride + 1

    windows = []
    for i in range(num_windows):
        start_index = i * win_stride
        end_index = start_index + win_size
        # Check if the end index goes beyond the array length
        if end_index > len(x):
            break  # Break the loop if the window exceeds the array length
        window = x[start_index:end_index]
        windows.append(window)

    # Convert the list of windows to a numpy array
    windows_array = np.array(windows)

    return windows_array # this returns an array where each element is a window of data -> for feature extraction




def slidingWindowGlove(x, win_size, win_stride):


    num_windows = 1 + (len(x) - win_size) // win_stride

    windows = []

    for i in range(num_windows):
        start_index = i * win_stride
        end_index = start_index + win_size
        end_segment_window = x[start_index:end_index] # not working
        window_mean = np.mean(end_segment_window)
        windows.append(window_mean)

    # convert the list of windows to a numpy array
    windows_array = np.array(windows) # this returns an array where each element is the final data point for that window 

    return windows_array


def WL(windows_array, win_size, win_stride):

    windows_WL_list = []

    for window in windows_array:
        window_WL = 1/win_size * np.sum(np.abs(np.diff(window)))
        windows_WL_list.append(window_WL)
    
    windows_WL_array = np.array(windows_WL_list) # should be one dimensional (num_windows,)

    return windows_WL_array

def LV(windows_array, win_size):
    windows_LV_list = []
    for window in windows_array:

        # log smoothing needed 
        epsilon = 1e-15

        window_LV = np.log10(np.var(window)+epsilon) # 
        windows_LV_list.append(window_LV)
    windows_LV_array = np.array(windows_LV_list)

    return windows_LV_array


def slidingWindowParameters(frequency, size_secs, stride_secs):
    # NB: size and stride should be in Seconds (150ms = 150 * 10**(-3))
    window_size = frequency * size_secs
    window_stride = frequency * stride_secs

    return window_size, window_stride

## --------- END SLIDING WINDOW FUNCTIONS ---------------- #


### ------- EMG , GLOVE AND RESTIMULUS PROCESSING START ----- ###

def emg_process(size_val, stride_val, user, dataset):

    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    fs = 2000

    size, stride = slidingWindowParameters(2000, (size_val*10**(-3)), (stride_val*10**(-3)))

    size = int(size)
    stride = int(stride)

    # 2. get the data: input is user and which dataset 

    emg_temp, glove_temp, stim_temp = getdata(user = user, dataset = dataset)

    # 3. low-pass filter the EMG data and extract features

    cutoff_val = 450
    order = 2

    for i in range(emg_temp.shape[1]):
        emg_temp[:, i] = lowpass_filter(emg_temp[:, i], cutoff_val, fs, order)

    del glove_temp, stim_temp

    emg_windows = []

    # Loop through each channelc
    num_channelsEMG = emg_temp.shape[1] 


    # decide which features to be extracted: 
    WL_bool = True
    LV_bool = True

    # low pass and features -> 

    for i in range(num_channelsEMG):
        
        
        emg_window = slidingWindowEMG(emg_temp[:, i], size, stride)

        if WL_bool: 
            emg_window_WL = WL(emg_window, size, stride) 
            emg_windows.append(emg_window_WL)

        if LV_bool:
            emg_window_LV = LV(emg_window, size)
            emg_windows.append(emg_window_LV)


        emg_windows_stacked = np.array(emg_windows)
        emg_windows_stacked = np.transpose(emg_windows_stacked)

        scaler = StandardScaler()
        emg_windows_stacked = scaler.fit_transform(emg_windows_stacked)

        directory = f'./processed_data/emg/dataset{dataset}'

        ensure_directory_exists(directory)


        emg_data_ID = f"{user}_{dataset}"

        
        np.save(f"{directory}/emg_{emg_data_ID}.npy", emg_windows_stacked)


    return emg_data_ID



def glove_process(size_val, stride_val, user, dataset):

    """
    The transformation matrix is from: 

    Krasoulis, Agamemnon, Sethu Vijayakumar, and Kianoush Nazarpour. 
    "Effect of user practice on prosthetic finger control with an 
    intuitive myoelectric decoder." Frontiers in neuroscience 13 (2019): 461612.

    https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00891/full#supplementary-material

    """

    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    emg_temp, glove_temp, stim_temp = getdata(user = user, dataset = dataset)
    del emg_temp, stim_temp
    
    size, stride = slidingWindowParameters(2000, (size_val*10**(-3)), (stride_val*10**(-3)))
    size = int(size)
    stride = int(stride)


    # perform sliding windows on the data
        
    glove_data = []

    for i in range(glove_temp.shape[1]):

        glove_data.append(slidingWindowGlove(glove_temp[:, i], size, stride))

    
    glove_data_array = np.array(glove_data).T


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

    A = A_T.T

    glove_mapped = A @ glove_data_array.T

    # get into expected sklearn input X.shape should be [number_of_samples, number_of_features] for scaling

    glove_mapped = glove_mapped.T

    # check that the shape is correct for sklearn scaling:
    if not glove_mapped.shape[1] == 5:
        raise ValueError("DoA is equal to 5, number of features for sklearn func should be 5. Check shape of data.")

    scaler = StandardScaler()
    glove_data_processed = scaler.fit_transform(glove_mapped)

    directory = f'./processed_data/glove/dataset{dataset}'

    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


    ensure_directory_exists(directory)
    glove_data_ID = f"{user}_{dataset}"
    np.save(f"{directory}/glove_{glove_data_ID}.npy", glove_data_processed)


    return glove_data_ID





### ------- EMG , GLOVE AND RESTIMULUS PROCESSING END ----- ###
