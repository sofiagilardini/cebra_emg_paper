import numpy as np
import matplotlib.pyplot as plt
import cebra 
import aux_functions as auxf
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import umap
from autoencoders_skorch import AutoEncoder_1, AutoEncoder_2, AutoEncoderNet, VariationalAutoEncoder, VariationalAutoEncoderNet
import pandas as pd
from sklearn.metrics import r2_score


"""
The goal of this script is to use the predictions for the DoA that has already been generated (for CEBRA, PCA, 
UMAP, Autoencoder and raw) and calculate the r_sq values. 

"""

def calcRsq(
        user: int, 
        dimred_type: str, 
        dimredID: str, 
        dataset: int):
    

    # load the MLP predictions 

    y_hat = auxf.getMLP_pred(
        user = user, 
        dimred_type= dimred_type, 
        dimred_ID=dimredID, 
        dataset=dataset
    )

    print(y_hat.shape)

    y = auxf.getProcessedData(
        user = user, 
        dataset=dataset, 
        mode = 'glove'
    )

    r2_list = []

    # calculate the multivariate r2 
    r2_list.append(r2_score(y_true = y, y_pred=y_hat, multioutput='uniform_average'))

    # check the shape is correct:

    if not y.shape[1] == 5 and not y_hat.shape[1] == 5:
        raise ValueError("Shape not correctly defined")

    for i in range(5):
        r2_list.append(r2_score(y_true = y[:, i], y_pred= y_hat[:, i]))

    return r2_list




# doar2 = calcRsq(user = 1, dimred_type='cebra_t', dimredID='offset5-model_0_1', dataset=1)

# r2_results = []

# r2_results.append(doar2)

# reg_results_df = pd.DataFrame(r2_results, columns = ['mvr2', 'DoA1', 'DoA2', 'DoA3', 'DoA4', 'DoA5'])

# breakpoint()


# ---- GLOBAL ------- #

user_list = np.arange(1,13)
dataset_list = [1, 2, 3]
MLP_struct = (100, 120, 100) # ? 
iters_MLP = 5 # ?
iters_CEBRA = 2
dataset_trainval = [1, 2]
size_val = 128
stride_val = 50


# ------- CEBRA ------- #

model_arch_list = ['offset5-model', 'offset10-model', 'offset36-model']
min_temp_list = [0, 0.2, 0.4, 0.6, 0.8]
cebra_modal_list = ['cebra_b', 'cebra_h', 'cebra_t']

time_offset_dict = {
    "offset5-model" : [1, 2, 4], 
    "offset10-model" : [2, 4, 8],
    "offset36-model" : [2, 10, 20, 32]
}


r2_results = []

for user in user_list:
    for model_arch in model_arch_list:
        for cebra_modal in cebra_modal_list:
            for min_temp_val in min_temp_list:

                offset_list = time_offset_dict[model_arch]

                for time_offset_val in offset_list:

                    dimred_ID = f"{model_arch}_{min_temp_val}_{time_offset_val}"
                    doa_list = calcRsq(user = user, dimred_type= cebra_modal, dimredID=dimred_ID, dataset=1)
                    doa_list.append(dimred_ID)
                    r2_results.append(doa_list)
                    print(dimred_ID)


reg_results_df = pd.DataFrame(r2_results, columns = ['mvr2', 'DoA1', 'DoA2', 'DoA3', 'DoA4', 'DoA5', 'dimred_ID'])

breakpoint()
