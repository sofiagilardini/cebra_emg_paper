
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
        window = x[start_index:end_index]

        # Calculate start index for the last 40% of the window #updated -- the whole window 
        mean_start_index = start_index + int(win_size)

        end_segment_window = x[mean_start_index:end_index]

        window_mean = np.mean(end_segment_window)

        windows.append(window_mean)

    # convert the list of windows to a numpy array
        
    windows_array = np.array(windows) # this returns an array where each element is the final data point for that window 

    return windows_array


def slidingWindowRestimulus(x, win_size, win_stride):


    num_windows = 1 + (len(x) - win_size) // win_stride

    windows = []

    for i in range(num_windows):
        start_index = i * win_stride
        end_index = start_index + win_size
        window = x[start_index:end_index]

        # Calculate the mode of the window
        mode_result = stats.mode(window)
        
        if np.isscalar(mode_result.mode):
            window_mode = mode_result.mode
        else:
            window_mode = mode_result.mode[0]
        
        windows.append(window_mode)

    # convert the list of windows to a numpy array
        
    windows_array = np.array(windows) # this returns an array where each element is the
    # final data point for that window 

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


def RMS(windows_array, win_size, win_stride):

    windows_RMS_list = []

    for window in windows_array:

        window_RMS = np.sqrt(np.mean(np.square(window)))
        windows_RMS_list.append(window_RMS) 
    windows_RMS_array = np.array(windows_RMS_list)

    return windows_RMS_array


def slidingWindowParameters(frequency, size_secs, stride_secs):
    # NB: size and stride should be in Seconds (150ms = 150 * 10**(-3))
    window_size = frequency * size_secs
    window_stride = frequency * stride_secs

    return window_size, window_stride

## --------- END SLIDING WINDOW FUNCTIONS ---------------- #


### ------- EMG , GLOVE AND RESTIMULUS PROCESSING START ----- ###

def emg_process(cutoff_val, size_val, stride_val, user, dataset, order):
        
    feat_ID = 'all'

    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    if feat_ID == 'all':
        raw_bool = False
        WL_bool = True
        LV_bool = True
        RMS_bool = True
        # SSC_bool = True

    
    fs = 2000

    size, stride = slidingWindowParameters(2000, (size_val*10**(-3)), (stride_val*10**(-3)))

    size = int(size)
    stride = int(stride)
    # 2. get the data: input is user and which dataset 

    emg_u1_d1, glove_u1_d1, stimulus_u1_d1 = getdata(user = user, dataset = dataset)

    # 3. low-pass filter the EMG data and extract features

    for i in range(emg_u1_d1.shape[1]):
        emg_u1_d1[:, i] = lowpass_filter(emg_u1_d1[:, i], cutoff_val, fs, order)

    del glove_u1_d1, stimulus_u1_d1

    if raw_bool: 


        # standardise the data 

        scaler = StandardScaler()
        emg_u1_d1 = scaler.fit_transform(emg_u1_d1)

        emg_data_ID = f"{user}_{dataset}"

        directory = f"./processed_data/emg_raw/dataset{dataset}"

        ensure_directory_exists(directory)
    
        np.save(f"{directory}/emg_{emg_data_ID}.npy", emg_u1_d1)


    if not raw_bool:

        emg_windows = []

        # Loop through each channelc
        num_channelsEMG = emg_u1_d1.shape[1] 


        # low pass and features -> 

        for i in range(num_channelsEMG):
            
            
            emg_window = slidingWindowEMG(emg_u1_d1[:, i], size, stride)

            if WL_bool: 
                emg_window_WL = WL(emg_window, size, stride) 
                emg_windows.append(emg_window_WL)



            if LV_bool:
                emg_window_LV = LV(emg_window, size)
                emg_windows.append(emg_window_LV)



            if RMS_bool:
                emg_window_RMS = RMS(emg_window, size, stride)
                emg_windows.append(emg_window_RMS)



            # emg_window_SSC = SSC(emg_window, size, stride)
            # SSC_bool = True
            # print("SSC", emg_window_SSC)

            # emg_windows.append(emg_window_SSC)


        emg_windows_stacked = np.array(emg_windows)
        emg_windows_stacked = np.transpose(emg_windows_stacked)

        scaler = StandardScaler()
        emg_windows_stacked = scaler.fit_transform(emg_windows_stacked)


        directory = f'./processed_data/emg/dataset{dataset}'

        ensure_directory_exists(directory)


        emg_data_ID = f"{user}_{dataset}_{cutoff_val}_{size_val}_{stride_val}_{feat_ID}"

        
        np.save(f"{directory}/emg_{emg_data_ID}.npy", emg_windows_stacked)



    return emg_data_ID



def glove_process(size_val, stride_val, user, dataset, rawBool: bool):

    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    emg_u1_d1, glove_u1_d1, stimulus_u1_d1 = getdata(user = user, dataset = dataset)

    del emg_u1_d1, stimulus_u1_d1

    if rawBool == True:  

        scaler = StandardScaler()
        glove_u1_d1 = scaler.fit_transform(glove_u1_d1)

        glove_data_ID = f"{user}_{dataset}"

        directory = f"./processed_data/glove_raw/dataset{dataset}"

        ensure_directory_exists(directory)
    
        np.save(f"{directory}/glove_{glove_data_ID}.npy", glove_u1_d1)



    if rawBool == False:
    
        size, stride = slidingWindowParameters(2000, (size_val*10**(-3)), (stride_val*10**(-3)))

        size = int(size)
        stride = int(stride)
        

        # perform sliding windows on the data
            
        glove_data = []

        for i in range(glove_u1_d1.shape[1]):

            glove_data.append(slidingWindowGlove(glove_u1_d1[:, i], size, stride))

        
        glove_data_array = np.array(glove_data).T

        scaler = StandardScaler()
        glove_data_array = scaler.fit_transform(glove_data_array)


        directory = f'./processed_data/glove/dataset{dataset}'

        def ensure_directory_exists(directory):
            if not os.path.exists(directory):
                os.makedirs(directory)


        ensure_directory_exists(directory)


        glove_data_ID = f"{user}_{dataset}_{size_val}_{stride_val}"

        
        np.save(f"{directory}/glove_{glove_data_ID}.npy", glove_data_array)




    return glove_data_ID




def restimulusProcess(size_val, stride_val, user, dataset, rawBool: bool):


    emg_u1_d1, glove_u1_d1, restimulus_u1_d1 = getdata(user = user, dataset = dataset)

    del emg_u1_d1, glove_u1_d1

    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


    if rawBool == True:     

        restimulus_data_ID = f"{user}_{dataset}"

        directory = f"./processed_data/restimulus_raw/dataset{dataset}"

        ensure_directory_exists(directory)
    
        np.save(f"{directory}/restimulus_{restimulus_data_ID}.npy", restimulus_u1_d1.astype(int))


    if rawBool == False:
    
        size, stride = slidingWindowParameters(2000, (size_val*10**(-3)), (stride_val*10**(-3)))

        size = int(size)
        stride = int(stride)
        

        restimulus_data = []

        for i in range(restimulus_u1_d1.shape[1]):
            restimulus_data.append(slidingWindowRestimulus(restimulus_u1_d1[:, i], size, stride))

        restimulus_data_array = np.array(restimulus_data).T

        directory = f'./processed_data/restimulus/dataset{dataset}'

        def ensure_directory_exists(directory):
            if not os.path.exists(directory):
                os.makedirs(directory)


        ensure_directory_exists(directory)


        restimulus_data_ID = f"{user}_{dataset}_{size_val}_{stride_val}"

        
        np.save(f"{directory}/restimulus_{restimulus_data_ID}.npy", restimulus_data_array.astype(int))



    return restimulus_data_ID




### ------- EMG , GLOVE AND RESTIMULUS PROCESSING END ----- ###
