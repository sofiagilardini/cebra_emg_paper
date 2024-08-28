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
import csv



# import regression_gridsearch 


"""
The goal of this script is to use the predictions for the DoA that has already been generated (for CEBRA, PCA, 
UMAP, Autoencoder and raw) and calculate the r_sq values. 

"""

dataset_trainval = [1, 2]

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


# ---- GLOBAL ------- #

user_list = np.arange(1,13)
dataset_list = [1, 2, 3]
MLP_struct = (100, 100, 100) # ? 
# iters_MLP = 200
# iters_CEBRA = 10000
dataset_trainval = [1, 2]
# size_val = 128
# stride_val = 50


# ------- CEBRA ------- #

model_arch_list = ['offset10-model', 'offset36-model']
min_temp_list = [0.05, 0.5, 0.8]
cebra_modal_list = ['cebra_b', 'cebra_h', 'cebra_t']

time_offset_dict = {
    "offset5-model" : [2, 4], 
    "offset10-model" : [4, 8],
    "offset36-model" : [20, 32]
}


pca_r2_results = []

for user in user_list: 
    for dataset in dataset_trainval:

        r2_list_temp = [] 

        r2_list_temp = calcRsq(user = user,
                dimred_type='PCA', 
                dimredID='PCA', 
                dataset = dataset)
        
        # r2_list_temp.append(user)
        # r2_list_temp.append(dataset)

        r2_list_temp.extend([user, dataset])
    
        pca_r2_results.append(r2_list_temp)

pca_results_df = pd.DataFrame(pca_r2_results , columns = ['mvr2', 'DoA1', 'DoA2', 'DoA3', 'DoA4', 'DoA5', 'user', 'dataset'])
pca_results_path = './results_df/all'
auxf.ensure_directory_exists(pca_results_path)
pca_results_df.to_csv(f"{pca_results_path}/pca_results_all.csv", index = False)


r2_results = []

for dataset in dataset_trainval:
    for cebra_modal in cebra_modal_list:
        for user in user_list:
            for model_arch in model_arch_list:
                for min_temp_val in min_temp_list:

                    offset_list = time_offset_dict[model_arch]

                    for time_offset_val in offset_list:

                        dimred_ID = f"{model_arch}_{min_temp_val}_{time_offset_val}"
                        doa_list = calcRsq(user = user, dimred_type= cebra_modal, dimredID=dimred_ID, dataset=dataset)
                        # doa_list.append(user)
                        # doa_list.append(dimred_ID)
                        # doa_list.append(cebra_modal)
                        # doa_list.append(model_arch)
                        # doa_list.append(min_temp_val)
                        # doa_list.append(time_offset_val)
                        # doa_list.append(dataset)

                        doa_list.extend([user, dimred_ID, cebra_modal, model_arch, min_temp_val, time_offset_val, dataset])

                        r2_results.append(doa_list)

reg_results_df = pd.DataFrame(r2_results, columns = ['mvr2', 'DoA1', 'DoA2', 'DoA3', 'DoA4', 'DoA5', 'user', 'dimred_ID', 'cebra_modal', 'model_arch', 'min_temp_val', 'time_offset_val', 'dataset'])

df_path = f"./results_df/all"
auxf.ensure_directory_exists(df_path)
reg_results_df.to_csv(f"{df_path}/r2_all_cebra_df.csv", index = False)



# n_neighbours_list = [5, 10, 20, 100, 200]
# min_dist_list = [0, 0.1, 0.25, 0.5, 0.8]

n_neighbours_list = [5, 200]
min_dist_list = [0.1, 0.8]

r2_results = []

for dataset in dataset_trainval:
    for user in user_list:
        for n_neighbors in n_neighbours_list:
            for min_dist in min_dist_list:
                dimred_type = "UMAP"
                dimred_ID = f"{dimred_type}_{n_neighbors}_{min_dist}"

                doa_list = calcRsq(user = user, dimred_type= dimred_type, dimredID=dimred_ID, dataset=dataset)
                # doa_list.append(user)
                # doa_list.append(dimred_ID)
                # doa_list.append(dataset)

                doa_list.extend([user, dimred_ID, dataset])


                r2_results.append(doa_list)

reg_results_df = pd.DataFrame(r2_results, columns = ['mvr2', 'DoA1', 'DoA2', 'DoA3', 'DoA4', 'DoA5', 'user', 'dimred_ID', 'dataset'])

df_path = f"./results_df/all"
auxf.ensure_directory_exists(df_path)
reg_results_df.to_csv(f"{df_path}/r2_{dimred_type}_df.csv", index = False)



# create UMAP results heatmap for n_neighbours and min_dist: for dataset 2!

    # create a new 'summary' df using the hyperparams

hyperparam_results = []

for n_neighbors in n_neighbours_list:
    for min_dist in min_dist_list:

        hyperparam_temp_list = []

        dimred_type = "UMAP"
        dimred_ID = f"{dimred_type}_{n_neighbors}_{min_dist}"

        df = pd.read_csv(f"./results_df/all/r2_{dimred_type}_df.csv")

        df_dataset2 = df[df['dataset'] == 2]

        mv_rsq_mean = df_dataset2['mvr2'][df_dataset2['dimred_ID'] == dimred_ID].mean()
        mv_rsq_var = df_dataset2['mvr2'][df_dataset2['dimred_ID'] == dimred_ID].var()

        hyperparam_temp_list = [
            dimred_ID,
            n_neighbors, 
            min_dist, 
            mv_rsq_mean, 
            mv_rsq_var
        ]

        hyperparam_results.append(hyperparam_temp_list)

summary_df = pd.DataFrame(hyperparam_results, columns=['dimred_ID', 'n_neighbors', 'min_dist', 'mvr2_mean', 'mvr2_var'])

df_path = f"./results_df/summary_validation"
auxf.ensure_directory_exists(df_path)
summary_df.to_csv(f"{df_path}/r2_{dimred_type}_df.csv", index = False)


# ------------ Autoencoder ---------------- # 


ae_r2_results = []

for user in user_list: 
    for dataset in dataset_trainval:

        r2_list_temp = [] 

        r2_list_temp = calcRsq(user = user,
                dimred_type='autoencoder', 
                dimredID='autoencoder', 
                dataset = dataset)
        
        # r2_list_temp.append(user)
        # r2_list_temp.append(dataset)

        r2_list_temp.extend([user, dataset])
    
        ae_r2_results.append(r2_list_temp)

ae_results_df = pd.DataFrame(ae_r2_results , columns = ['mvr2', 'DoA1', 'DoA2', 'DoA3', 'DoA4', 'DoA5', 'user', 'dataset'])
ae_results_path = './results_df/all'
auxf.ensure_directory_exists(ae_results_path)
ae_results_df.to_csv(f"{ae_results_path}/autoencoder_results_all.csv", index = False)

# -------------------------------------------------------- #

# ------------ no_dimred ---------------- # 


nodimred_r2_results = []

for user in user_list: 
    for dataset in dataset_trainval:

        r2_list_temp = [] 

        r2_list_temp = calcRsq(user = user,
                dimred_type='no_dimred', 
                dimredID='no_dimred', 
                dataset = dataset)
        
        # r2_list_temp.append(user)
        # r2_list_temp.append(dataset)
    
        r2_list_temp.extend([user, dataset])

        nodimred_r2_results.append(r2_list_temp)

nodimred_results_df = pd.DataFrame(nodimred_r2_results , columns = ['mvr2', 'DoA1', 'DoA2', 'DoA3', 'DoA4', 'DoA5', 'user', 'dataset'])
nodimred_results_path = './results_df/all'
auxf.ensure_directory_exists(ae_results_path)
nodimred_results_df.to_csv(f"{ae_results_path}/no_dimred_results_all.csv", index = False)

# -------------------------------------------------------- #


id_results = []

for cebra_modal in cebra_modal_list:
    for model_arch in model_arch_list:
        for min_temp_val in min_temp_list:

            offset_list = time_offset_dict[model_arch]

            for time_offset_val in offset_list:

                dimred_ID = f"{model_arch}_{min_temp_val}_{time_offset_val}"

                df = pd.read_csv(f"./results_df/all/r2_all_cebra_df.csv")

                df_dataset2 = df[df['dataset'] == 2]

                df_dataset2 = df[df['cebra_modal'] == cebra_modal]

                id_mean = df_dataset2['mvr2'][df_dataset2['dimred_ID'] == dimred_ID].mean()
                id_var = df_dataset2['mvr2'][df_dataset2['dimred_ID'] == dimred_ID].var()

                id_list = [
                    dimred_ID,
                    cebra_modal, 
                    model_arch, 
                    min_temp_val, 
                    time_offset_val, 
                    id_mean, 
                    id_var
                ]

                id_results.append(id_list)


reg_results_df = pd.DataFrame(id_results, columns = ['dimred_ID', 'cebra_modal', 'model_arch', 'min_temp_val', 'time_offset_val', 'mvr2_mean', 'mvr2_var'])

df_path = f"./results_df/summary_validation"
auxf.ensure_directory_exists(df_path)
reg_results_df.to_csv(f"{df_path}/r2_cebra_hyperp_summary.csv", index = False)


# find the best hyperparam combination for each cebra modality

best_cebra_results = []
summary_results = pd.read_csv('./results_df/summary_validation/r2_cebra_hyperp_summary.csv')


for cebra_modal in cebra_modal_list:

    best_cebra_results_temp = []

    filtered_results = summary_results[summary_results['cebra_modal'] == cebra_modal]


    best_hyperparms_ind = filtered_results['mvr2_mean'].idxmax()

    best_values = filtered_results.loc[best_hyperparms_ind]

    # best_cebra_results_temp = [
    #     cebra_modal, 
    #     filtered_results['dimred_ID'][best_hyperparms_ind],
    #     filtered_results['model_arch'][best_hyperparms_ind],
    #     filtered_results['min_temp_val'][best_hyperparms_ind],
    #     filtered_results['time_offset_val'][best_hyperparms_ind],
    #     filtered_results['mvr2_mean'][best_hyperparms_ind],
    #     filtered_results['mvr2_var'][best_hyperparms_ind],
    # ]

    best_cebra_results_temp = [
    cebra_modal, 
    best_values['dimred_ID'],
    best_values['model_arch'],
    best_values['min_temp_val'],
    best_values['time_offset_val'],
    best_values['mvr2_mean'],
    best_values['mvr2_var'],
]

    best_cebra_results.append(best_cebra_results_temp)


best_results_df = pd.DataFrame(best_cebra_results, columns = ['cebra_modal', 'dimred_ID', 'model_arch', 'min_temp_val', 'time_offset_val', 'mvr2_mean', 'mvr2_var'])

df_path = f"./results_df/best_validation"
auxf.ensure_directory_exists(df_path)
best_results_df.to_csv(f"{df_path}/r2_cebra_hyperp_best.csv", index = False)

# find the best UMAP hyperparams 


umap_df = pd.read_csv('./results_df/summary_validation/r2_UMAP_df.csv')

max_mvr2_mean_indx = umap_df['mvr2_mean'].idxmax()

umap_best_hyperp = []

umap_best_hyp_temp = [
    umap_df['dimred_ID'][max_mvr2_mean_indx],
    umap_df['n_neighbors'][max_mvr2_mean_indx],
    umap_df['min_dist'][max_mvr2_mean_indx],
    umap_df['mvr2_mean'][max_mvr2_mean_indx],
    umap_df['mvr2_var'][max_mvr2_mean_indx]
]

umap_best_hyperp.append(umap_best_hyp_temp)


umap_best_hyp_df = pd.DataFrame(umap_best_hyperp, columns = ['dimred_ID', 'n_neighbors', 'min_dist', 'mvr2_mean', 'mvr2_var'])


df_path = f"./results_df/best_validation"
auxf.ensure_directory_exists(df_path)
umap_best_hyp_df.to_csv(f"{df_path}/r2_umap_hyperp_best.csv", index = False)





# start generating 'total' results df - based on performance on dataset 2 (validation)

cebra_df = pd.read_csv('./results_df/best_validation/r2_cebra_hyperp_best.csv')
umap_df = pd.read_csv('./results_df/summary_validation/r2_UMAP_df.csv')
pca_df = pd.read_csv('./results_df/all/pca_results_all.csv')
ae_df = pd.read_csv('./results_df/all/autoencoder_results_all.csv')
no_dimred_df = pd.read_csv('./results_df/all/no_dimred_results_all.csv')

umap_df_filtered = umap_df[umap_df['mvr2_mean'] == umap_df['mvr2_mean'].max()]

cebra_b = cebra_df[cebra_df['cebra_modal'] == 'cebra_b'] # do not need to separate d1 and d2 because this is the validation df
cebra_h = cebra_df[cebra_df['cebra_modal'] == 'cebra_h']
cebra_t = cebra_df[cebra_df['cebra_modal'] == 'cebra_t']

pca_d2 = pca_df[pca_df['dataset'] == 2]
ae_d2 = ae_df[ae_df['dataset'] == 2]
no_dimred_d2 = no_dimred_df[no_dimred_df['dataset'] == 2]


all_dimred_results = [
    [cebra_b['cebra_modal'].values[0], cebra_b['mvr2_mean'].values[0], cebra_b['mvr2_var'].values[0]],
    [cebra_h['cebra_modal'].values[0], cebra_h['mvr2_mean'].values[0], cebra_h['mvr2_var'].values[0]],
    [cebra_t['cebra_modal'].values[0], cebra_t['mvr2_mean'].values[0], cebra_t['mvr2_var'].values[0]],
    ["UMAP", umap_df_filtered['mvr2_mean'].values[0], umap_df_filtered['mvr2_var'].values[0]], 
    ["PCA", pca_d2['mvr2'].mean(), pca_d2['mvr2'].var()], 
    ["Autoencoder", ae_d2['mvr2'].mean(), ae_d2['mvr2'].var()], 
    ["No DimRed", no_dimred_d2['mvr2'].mean(), no_dimred_d2['mvr2'].var()]]

results_path = './results_df/summary_validation'
auxf.ensure_directory_exists(results_path)

all_dimred_results_df = pd.DataFrame(all_dimred_results, columns = ['dimred', 'mvr2_mean', 'mvr2_var'])
all_dimred_results_df.to_csv(f"{results_path}/validation_all_dimred_results.csv", index = False)


# ------------- get training and testing results for the best hyperparams found using validation ------------- #


# test first

# CEBRA

test_r2_results = []


cebra_best_df = pd.read_csv('results_df/best_validation/r2_cebra_hyperp_best.csv')

for cebra_modal in cebra_modal_list:

    cebra_df_filtered = cebra_best_df[cebra_best_df['cebra_modal'] == cebra_modal]

    model_arch = cebra_df_filtered['model_arch'].values[0]
    min_temp_val = cebra_df_filtered['min_temp_val'].values[0]
    time_offset_val = cebra_df_filtered['time_offset_val'].values[0]

    dimred_ID = f"{model_arch}_{min_temp_val}_{time_offset_val}"

    # trainval

    for user in user_list:

        dataset = 3

        emg = np.load(f"./processed_data/user{user}/emg/dataset{dataset}/emg_{user}_{dataset}.npy")
        glove = np.load(f"./processed_data/user{user}/glove/dataset{dataset}/glove_{user}_{dataset}.npy")

        cebra_model_dir = f'./cebra_models/user{user}/{cebra_modal}'
        cebra_model_ID = f"{model_arch}_{min_temp_val}_{time_offset_val}"


        cebra_model = cebra.CEBRA.load(f"{cebra_model_dir}/{cebra_model_ID}.pt")

        embedding = cebra_model.transform(emg)


        embedding_dir = f'./embeddings/user{user}/{cebra_modal}/dataset{dataset}'
        auxf.ensure_directory_exists(embedding_dir)

        np.save(f"{embedding_dir}/{cebra_model_ID}.npy", embedding)

        MLP_dir = f'./trained_MLP/user{user}/{cebra_modal}'
        MLP_path = f"{MLP_dir}/{cebra_model_ID}.pkl"

        trained_MLP = joblib.load(MLP_path)

        emg_embedding_ = np.load(f"./embeddings/user{user}/{cebra_modal}/dataset{dataset}/{cebra_model_ID}.npy")

        pred_path = f"./MLP_pred/user{user}/{cebra_modal}/dataset{dataset}"
        auxf.ensure_directory_exists(pred_path)


        pred_d1 = trained_MLP.predict(emg_embedding_)
        np.save(f"{pred_path}/{cebra_model_ID}", pred_d1)

        r2_list = calcRsq(
        user = user, 
        dimred_type = cebra_modal, 
        dimredID = dimred_ID, 
        dataset = dataset)
    
        r2_list.append(user)
        r2_list.append(dataset)
        r2_list.append(cebra_modal)
        
        test_r2_results.append(r2_list)



# UMAP

umap_df = pd.read_csv('results_df/best_validation/r2_umap_hyperp_best.csv')

n_neighbors = umap_df['n_neighbors'].values[0]
min_dist = umap_df['min_dist'].values[0]

dimred_type = "UMAP"
dimred_ID = f"{dimred_type}_{n_neighbors}_{min_dist}"

MLP_dir = f'./trained_MLP/user{user}/{dimred_type}'
MLP_path = f"{MLP_dir}/{dimred_ID}.pkl"

trained_MLP = joblib.load(MLP_path)

dataset = 3


for user in user_list:

    emg_embedding_d3 = np.load(f"./embeddings/user{user}/{dimred_type}/dataset{dataset}/{dimred_ID}.npy")
    glove_d3 = auxf.getProcessedData(user = user,
                                        dataset=dataset, 
                                        mode = 'glove')

    pred_path = f"./MLP_pred/user{user}/{dimred_type}/dataset{dataset}"
    auxf.ensure_directory_exists(pred_path)

    pred_d3 = trained_MLP.predict(emg_embedding_d3)
    np.save(f"{pred_path}/{dimred_ID}.npy", pred_d3)

    r2_list = calcRsq(
        user = user, 
        dimred_type =  'UMAP', 
        dimredID = dimred_ID, 
        dataset = dataset)
    
    # r2_list.append(user)
    # r2_list.append(dataset)
    # r2_list.append(dimred_type)
    
    r2_list.extend([user, dataset, dimred_type])

    test_r2_results.append(r2_list)




# PCA 

dimred_type = "PCA"
dimred_ID = f"{dimred_type}"

MLP_dir = f'./trained_MLP/user{user}/{dimred_type}'
MLP_path = f"{MLP_dir}/{dimred_ID}.pkl"

trained_MLP = joblib.load(MLP_path)

dataset = 3

for user in user_list:

    emg_embedding_d3 = np.load(f"./embeddings/user{user}/{dimred_type}/dataset{dataset}/{dimred_ID}.npy")
    glove_d3 = auxf.getProcessedData(user = user,
                                        dataset=dataset, 
                                        mode = 'glove')

    pred_path = f"./MLP_pred/user{user}/{dimred_type}/dataset{dataset}"
    auxf.ensure_directory_exists(pred_path)

    pred_d3 = trained_MLP.predict(emg_embedding_d3)
    np.save(f"{pred_path}/{dimred_ID}.npy", pred_d3)

    r2_list = calcRsq(
        user = user, 
        dimred_type = 'PCA', 
        dimredID = dimred_ID, 
        dataset = dataset)
    
    # r2_list.append(user)
    # r2_list.append(dataset)
    # r2_list.append(dimred_type)

    r2_list.extend([user, dataset, dimred_type])

    test_r2_results.append(r2_list)


# Autoencoder

dimred_type = "autoencoder"
dimred_ID = f"{dimred_type}"

dataset = 3

for user in user_list:

    r2_list = calcRsq(
        user = user, 
        dimred_type = 'autoencoder', 
        dimredID = dimred_ID, 
        dataset = dataset)
    
    r2_list.append(user)
    r2_list.append(dataset)
    r2_list.append(dimred_type)
    
    test_r2_results.append(r2_list)


# No Dim Red


dimred_type = "no_dimred"
dimred_ID = f"{dimred_type}"

dataset = 3

for user in user_list:

    r2_list = calcRsq(
        user = user, 
        dimred_type = 'no_dimred', 
        dimredID = dimred_ID, 
        dataset = dataset)
    
    r2_list.append(user)
    r2_list.append(dataset)
    r2_list.append(dimred_type)
    
    test_r2_results.append(r2_list)


# save all TEST results

test_r2_results_df = pd.DataFrame(test_r2_results, columns = ['mvr2', 'DoA1', 'DoA2', 'DoA3', 'DoA4', 'DoA5', 'user', 'dataset', 'dimred_type'])
test_results_path = './results_df/test_results'
auxf.ensure_directory_exists(test_results_path)

test_r2_results_df.to_csv(f"{test_results_path}/test_results.csv", index = False)


# concatenate the test_results with all results to get a df with train val and test.

pca_df = pd.read_csv('./results_df/all/pca_results_all.csv')
pca_df['dimred_type'] = 'PCA'
ae_df = pd.read_csv('./results_df/all/autoencoder_results_all.csv')
ae_df['dimred_type'] = 'autoencoder'
no_dimred_df = pd.read_csv('./results_df/all/no_dimred_results_all.csv')
no_dimred_df['dimred_type'] = 'no_dimred'


cebra_hyp_df = pd.read_csv('./results_df/best_validation/r2_cebra_hyperp_best.csv')
umap_hyp_df = pd.read_csv('./results_df/best_validation/r2_umap_hyperp_best.csv')

hyp_ID = {
    'cebra_b' : cebra_hyp_df['dimred_ID'][cebra_hyp_df['cebra_modal'] == 'cebra_b'].values[0], 
    'cebra_h' : cebra_hyp_df['dimred_ID'][cebra_hyp_df['cebra_modal'] == 'cebra_h'].values[0],
    'cebra_t' : cebra_hyp_df['dimred_ID'][cebra_hyp_df['cebra_modal'] == 'cebra_t'].values[0],
    'UMAP' : umap_hyp_df['dimred_ID'].values[0]
}


cebra_trainval = pd.read_csv('./results_df/all/r2_all_cebra_df.csv')
cebra_trainval_filt = cebra_trainval.drop(columns = ['dimred_ID' , 'model_arch' , 'min_temp_val', 'time_offset_val'])

trainvaltest_df = pd.concat([test_r2_results_df, pca_df, ae_df, no_dimred_df])

trainvaltest_df = pd.concat([trainvaltest_df, 
                             cebra_trainval_filt[cebra_trainval['dimred_ID'] == hyp_ID['cebra_b']],
                             cebra_trainval_filt[cebra_trainval['dimred_ID'] == hyp_ID['cebra_h']],
                             cebra_trainval_filt[cebra_trainval['dimred_ID'] == hyp_ID['cebra_t']]               
                             ])


trainvaltest_df.to_csv("./results_df/trainvaltest.csv", index = False)