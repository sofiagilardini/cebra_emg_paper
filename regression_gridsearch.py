import numpy as np
import matplotlib.pyplot as plt
import cebra 
import aux_functions as auxf
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import umap
from autoencoders_skorch import AutoEncoder_1, AutoEncoder_2, AutoEncoderNet, VariationalAutoEncoder, VariationalAutoEncoderNet



params = {"font.family" : "serif"}
plt.rcParams.update(params)

np.random.seed(42)


def runCebraTraining(
        user: int, 
        model_arch: str, 
        cebra_modal: str, 
        min_temp_val: float, 
        time_offset_val: int):


    dataset = 1 # this should *not* change, dataset for training is always dataset1

    emg = np.load(f"./processed_data/user{user}/emg/dataset{dataset}/emg_{user}_{dataset}.npy")
    glove = np.load(f"./processed_data/user{user}/glove/dataset{dataset}/glove_{user}_{dataset}.npy")

    if cebra_modal == 'cebra_b':
        condit = 'time_delta'
        hybrid_bool = False

    elif cebra_modal == 'cebra_h':
        condit = 'time_delta'
        hybrid_bool = True

    elif cebra_modal == 'cebra_t':
        condit = 'time'
        hybrid_bool = False


    cebra_model = cebra.CEBRA(model_architecture= model_arch,
                            batch_size=512,
                            learning_rate=3e-4,
                            output_dimension=3,
                            max_iterations=iters_CEBRA,
                            distance='cosine',
                            device='cuda_if_available',
                            verbose=True,
                            conditional= condit,
                            time_offsets = time_offset_val,
                            min_temperature = min_temp_val, 
                            hybrid = hybrid_bool)



    if cebra_modal == 'cebra_b' or cebra_modal == 'cebra_h':
        cebra_model.fit(emg, glove)

    elif cebra_modal == 'cebra_t':
        cebra_model.fit(emg)

    cebra_model_dir = f'./cebra_models/user{user}/{cebra_modal}'
    auxf.ensure_directory_exists(cebra_model_dir)
    cebra_model_ID = f"{model_arch}_{min_temp_val}_{time_offset_val}"
    cebra_model.save(f"{cebra_model_dir}/{cebra_model_ID}.pt")



# using the CEBRA model trained on dataset1, produce embeddings for dataset1 and dataset2 

def runEmbeddings_CEBRA(user: int,
                  model_arch: str, 
                  cebra_modal: str, 
                  min_temp_val: float, 
                  time_offset_val: int):


    for dataset in dataset_trainval:

        emg = np.load(f"./processed_data/user{user}/emg/dataset{dataset}/emg_{user}_{dataset}.npy")
        glove = np.load(f"./processed_data/user{user}/glove/dataset{dataset}/glove_{user}_{dataset}.npy")

        cebra_model_dir = f'./cebra_models/user{user}/{cebra_modal}'
        cebra_model_ID = f"{model_arch}_{min_temp_val}_{time_offset_val}"

        cebra_model = cebra.CEBRA.load(f"{cebra_model_dir}/{cebra_model_ID}.pt")

        embedding = cebra_model.transform(emg)

        embedding_dir = f'./embeddings/user{user}/{cebra_modal}/dataset{dataset}'
        auxf.ensure_directory_exists(embedding_dir)

        np.save(f"{embedding_dir}/{cebra_model_ID}.npy", embedding)

    

def RunPredictions_CEBRA(
        user: int, 
        cebra_modal: str, 
        model_arch: str, 
        min_temp_val: float,
        time_offset_val: int):

    # first train all the models (dataset = 1)

    dataset = 1

    cebra_model_ID = f"{model_arch}_{min_temp_val}_{time_offset_val}"
    emg_embedding_d1 = np.load(f"./embeddings/user{user}/{cebra_modal}/dataset{dataset}/{cebra_model_ID}.npy")
    glove_d1 = np.load(f"./processed_data/user{user}/glove/dataset{dataset}/glove_{user}_{dataset}.npy")

    MLP_dir = f'./trained_MLP/user{user}/{cebra_modal}'
    auxf.ensure_directory_exists(MLP_dir)
    MLP_path = f"{MLP_dir}/{cebra_model_ID}.pkl"

    reg_MLP = MLPRegressor(hidden_layer_sizes = MLP_struct,
        max_iter = iters_MLP,
        shuffle = False,
        verbose = True)

    # fit MLP
    reg_MLP.fit(emg_embedding_d1, glove_d1)  
    # save MLP

    joblib.dump(reg_MLP, MLP_path) 
    # load MLP
    trained_MLP = joblib.load(MLP_path)

    # now we have the trained MLP - we can make predictions for dataset1, dataset2 

    for dataset in dataset_trainval:

        emg_embedding_ = np.load(f"./embeddings/user{user}/{cebra_modal}/dataset{dataset}/{cebra_model_ID}.npy")

        pred_path = f"./MLP_pred/user{user}/{cebra_modal}/dataset{dataset}"
        auxf.ensure_directory_exists(pred_path)

        pred_d1 = trained_MLP.predict(emg_embedding_)
        np.save(f"{pred_path}/{cebra_model_ID}", pred_d1)







def RunPredictions_(dimred_type: str, dimred_ID: str, user: int):

    """
    dimred_type = ['PCA', 'UMAP', 'Autoencoder']

    dimred_ID e.g : 'offset5-model_0_1'
    
    """

    # first train all the models (dataset = 1)

    MLP_dir = f'./trained_MLP/user{user}/{dimred_type}'
    auxf.ensure_directory_exists(MLP_dir)
    MLP_path = f"{MLP_dir}/{dimred_ID}.pkl"

    reg_MLP = MLPRegressor(hidden_layer_sizes = MLP_struct,
        max_iter = iters_MLP,
        shuffle = False,
        verbose = True)
    
    dataset = 1
    emg_embedding_d1 = np.load(f"./embeddings/user{user}/{dimred_type}/dataset{dataset}/{dimred_ID}.npy")
    glove_d1 = auxf.getProcessedData(user = user,
                                     dataset=dataset, 
                                     mode = 'glove')

    # fit MLP
    reg_MLP.fit(emg_embedding_d1, glove_d1)  
    # save MLP

    joblib.dump(reg_MLP, MLP_path) 
    # load MLP
    trained_MLP = joblib.load(MLP_path)

    pred_path = f"./MLP_pred/user{user}/{dimred_type}/dataset{dataset}"
    auxf.ensure_directory_exists(pred_path)

    pred_d1 = trained_MLP.predict(emg_embedding_d1)
    np.save(f"{pred_path}/{dimred_ID}.npy", pred_d1)

    # now we have the trained MLP - we can make predictions for dataset1, dataset2 

    dataset = 2
    emg_embedding_d2 = np.load(f"./embeddings/user{user}/{dimred_type}/dataset{dataset}/{dimred_ID}.npy")

    pred_path = f"./MLP_pred/user{user}/{dimred_type}/dataset{dataset}"
    auxf.ensure_directory_exists(pred_path)

    pred_d2 = trained_MLP.predict(emg_embedding_d2)
    np.save(f"{pred_path}/{dimred_ID}.npy", pred_d2)





def runEmbeddings_UMAP(user: int, 
                       n_neighbors: float, 
                       min_dist: float): 

    """
    keeping components = 3, and metric = 'cosine'

    n_neighbors: balance between maintaining global and local structure in the data. 
    - low values, concentrate on very local structure
    - [5, 10, 20, 100, 200]

    min_dist: how tightly UMAP is allowed to bring points together in the embedding 
    - low values of min_dist will result in clumpier embeddings
    - [0, 0.1, 0.25, 0.5, 0.8]
    
    """

    # constants 

    n_components = 3
    metric = 'cosine'
    dimred_type = 'UMAP'
    dimred_ID = f'{dimred_type}_{n_neighbors}_{min_dist}'

    # load user's emg data for dataset 1 to fit UMAP

    dataset = 1
    emg_d1 = auxf.getProcessedData(user = user, dataset = dataset, mode = 'emg')
    # fit UMAP on emg_d1

    umap_model = umap.UMAP(n_neighbors=n_neighbors, 
                           min_dist = min_dist, 
                           n_components=n_components, 
                           metric = metric)
    
    embedding = umap_model.fit_transform(emg_d1)

    embedding_dir = f'./embeddings/user{user}/{dimred_type}/dataset{dataset}'
    auxf.ensure_directory_exists(embedding_dir)

    np.save(f"{embedding_dir}/{dimred_ID}.npy", embedding)

    # transform dataset 2 and save

    dataset = 2
    emg_d2 = auxf.getProcessedData(user = user, dataset = dataset, mode = 'emg')

    embedding = umap_model.transform(emg_d2)
    embedding_dir = f'./embeddings/user{user}/{dimred_type}/dataset{dataset}'
    auxf.ensure_directory_exists(embedding_dir)

    np.save(f"{embedding_dir}/{dimred_ID}.npy", embedding)

    # length_ = np.arange(0, len(glove_d1))

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(embedding_d1[:, 0], embedding_d1[:, 1], embedding_d1[:, 2], cmap= "plasma", c = length_)
    # plt.show()


def runEmbeddings_PCA(user: int): 

    """
    keeping components = 3
    """

    # constants 

    dimred_type = 'PCA'
    n_components = 3
    dimred_ID = f'{dimred_type}'

    # load user's emg data for dataset 1 to fit UMAP

    dataset = 1
    emg_d1 = auxf.getProcessedData(user = user, dataset = dataset, mode = 'emg')
    # fit UMAP on emg_d1

    pca_model = PCA(n_components=3)
    
    embedding = pca_model.fit_transform(emg_d1)

    embedding_dir = f'./embeddings/user{user}/{dimred_type}/dataset{dataset}'
    auxf.ensure_directory_exists(embedding_dir)

    np.save(f"{embedding_dir}/{dimred_ID}.npy", embedding)

    # transform dataset 2 and save

    dataset = 2
    emg_d2 = auxf.getProcessedData(user = user, dataset = dataset, mode = 'emg')

    embedding = pca_model.transform(emg_d2)
    embedding_dir = f'./embeddings/user{user}/{dimred_type}/dataset{dataset}'
    auxf.ensure_directory_exists(embedding_dir)

    np.save(f"{embedding_dir}/{dimred_ID}.npy", embedding)




# ---- GLOBAL ------- #

user_list = np.arange(1,13)
dataset_list = [1, 2, 3]
MLP_struct = (100, 120, 100) # ? 
iters_MLP = 5 # ?
iters_CEBRA = 2
dataset_trainval = [1, 2]
size_val = 128
stride_val = 50

# # ------ DATA PROCESSING ----- # 

# auxf.runDataProcessing(
#     user_list = user_list, 
#     dataset_list=dataset_list, 
#     size_val=size_val, 
#     stride_val = stride_val)

# ------- CEBRA ------- #

model_arch_list = ['offset5-model', 'offset10-model', 'offset36-model']
min_temp_list = [0, 0.2, 0.4, 0.6, 0.8]
cebra_modal_list = ['cebra_b', 'cebra_h', 'cebra_t']

time_offset_dict = {
    "offset5-model" : [1, 2, 4], 
    "offset10-model" : [2, 4, 8],
    "offset36-model" : [2, 10, 20, 32]
}


for user in user_list:
    for model_arch in model_arch_list:
        for cebra_modal in cebra_modal_list:
            for min_temp_val in min_temp_list:

                offset_list = time_offset_dict[model_arch]

                for time_offset_val in offset_list:

                    dimred_ID = f"{cebra_modal}_{min_temp_val}_{time_offset_val}"

                    runCebraTraining(user = user, 
                                     model_arch=model_arch, 
                                     cebra_modal=cebra_modal, 
                                     min_temp_val=min_temp_val, 
                                     time_offset_val=time_offset_val)

                    runEmbeddings_CEBRA(user = user, 
                                        model_arch = model_arch, 
                                        cebra_modal = cebra_modal, 
                                        min_temp_val = min_temp_val,
                                        time_offset_val=time_offset_val)
                    
                    RunPredictions_CEBRA(user = user,
                                         cebra_modal=cebra_modal, 
                                         model_arch=model_arch, 
                                         min_temp_val=min_temp_val, 
                                         time_offset_val=time_offset_val)

# ------- CEBRA END ------- #

# # ----- UMAP -------- #


n_neighbours_list = [5, 10, 20, 100, 200]
min_dist_list = [0, 0.1, 0.25, 0.5, 0.8]


# for user in user_list:
#     for n_neighbors in n_neighbours_list:
#         for min_dist in min_dist_list:
#             dimred_type = "UMAP"
#             dimred_ID = f"{dimred_type}_{n_neighbors}_{min_dist}"

#             runEmbeddings_UMAP(user=user, 
#                             n_neighbors=n_neighbors, 
#                             min_dist=min_dist)
#             RunPredictions_(dimred_type = dimred_type, 
#                             dimred_ID=dimred_ID, 
#                             user = user)
            
# # # ----- UMAP END ------- #


# ---- PCA ------

for user in user_list:
    print("beginning PCA")
    dimred_type = "PCA"
    dimred_ID = dimred_type

    runEmbeddings_PCA(user=user)
    RunPredictions_(dimred_type = dimred_type, 
                    dimred_ID=dimred_ID, 
                    user = user)
    
    print("end PCA")

    
# ----- END PCA ------ #