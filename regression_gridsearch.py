import numpy as np
import matplotlib.pyplot as plt
import cebra 
import aux_functions as auxf
import joblib
from sklearn.neural_network import MLPRegressor


params = {"font.family" : "serif"}
plt.rcParams.update(params)

np.random.seed(42)

model_arch_list = ['offset5-model', 'offset10-model', 'offset36-model']
min_temp_list = [0, 0.2, 0.4, 0.6, 0.8]
cebra_modal_list = ['cebra_b', 'cebra_h', 'cebra_t']
user_list = [1]
dataset_list = [1, 2, 3]






# time_offset_dict = {
#     "offset5-model" : [1, 2, 4], 
#     "offset10-model" : [2, 4, 8],
#     "offset36-model" : [2, 10, 20, 32]
# }

time_offset_dict = {
    "offset5-model" : [1], 
    "offset10-model" : [1],
    "offset36-model" : [1]
}



def runTraining():

    iters = 2

    for user in user_list:

        dataset = 1 # this should not change, dataset for training is always 1

        emg = np.load(f"./processed_data/emg/dataset{dataset}/emg_{user}_{dataset}.npy")
        glove = np.load(f"./processed_data/glove/dataset{dataset}/glove_{user}_{dataset}.npy")


        for model_arch in model_arch_list:
            for cebra_modal in cebra_modal_list:
                for min_temp_val in min_temp_list:

                    offset_list = time_offset_dict[model_arch]

                    for time_offset_val in offset_list:

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
                                                max_iterations=iters,
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


    # ----------------- END OF ALL CEBRA MODEL TRAINING ----------------------- # 


# using the CEBRA model trained on dataset1, produce embeddings for dataset1, dataset2 and dataset 3

def runEmbeddings():

    user_list = [1]

    for user in user_list:

        dataset_list = [1, 2, 3]

        for dataset in dataset_list:
            emg = np.load(f"./processed_data/emg/dataset{dataset}/emg_{user}_{dataset}.npy")
            glove = np.load(f"./processed_data/glove/dataset{dataset}/glove_{user}_{dataset}.npy")

            for model_arch in model_arch_list:
                for cebra_modal in cebra_modal_list:
                    for min_temp_val in min_temp_list:

                        offset_list = time_offset_dict[model_arch]

                        for time_offset_val in offset_list:

                            cebra_model_dir = f'./cebra_models/user{user}/{cebra_modal}'
                            cebra_model_ID = f"{model_arch}_{min_temp_val}_{time_offset_val}"

                            cebra_model = cebra.CEBRA.load(f"{cebra_model_dir}/{cebra_model_ID}.pt")


                            embedding = cebra_model.transform(emg)
        
                            embedding_dir = f'./embeddings/user{user}/{cebra_modal}/dataset{dataset}'
                            auxf.ensure_directory_exists(embedding_dir)

                            np.save(f"{embedding_dir}/{cebra_model_ID}.npy", embedding)



def RunPredictions():

    MLP_struct = (100, 120, 100) # ? 
    iters_MLP = 5 # ?

    # first train all the models (dataset = 1)

    dataset = 1
    
    for user in user_list:
        for cebra_modal in cebra_modal_list:
            for model_arch in model_arch_list:
                for min_temp_val in min_temp_list:
                    
                    offset_list = time_offset_dict[model_arch]

                    for time_offset_val in offset_list:

                        cebra_model_ID = f"{model_arch}_{min_temp_val}_{time_offset_val}"
                        emg_embedding_d1 = np.load(f"./embeddings/user{user}/{cebra_modal}/dataset{dataset}/{cebra_model_ID}.npy")
                        glove_d1 = np.load(f"./processed_data/glove/dataset{dataset}/glove_{user}_{dataset}.npy")

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

                        # now we have the trained MLP - we can make predictions for dataset1, dataset2 and dataset3

                        for dataset in dataset_list:

                            emg_embedding_ = np.load(f"./embeddings/user{user}/{cebra_modal}/dataset{dataset}/{cebra_model_ID}.npy")

                            pred_path = f"./MLP_pred/user{user}/{cebra_modal}/dataset{dataset}"
                            auxf.ensure_directory_exists(pred_path)

                            pred_d1 = trained_MLP.predict(emg_embedding_)
                            np.save(f"{pred_path}/{cebra_model_ID}", pred_d1)



                        


RunPredictions()