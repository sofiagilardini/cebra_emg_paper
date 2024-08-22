import numpy as np
import matplotlib.pyplot as plt
import cebra 
import aux_functions as auxf



params = {"font.family" : "serif"}
plt.rcParams.update(params)

np.random.seed(42)

model_arch_list = ['offset5-model', 'offset10-model', 'offset36-model']
min_temp_list = [0, 0.2, 0.4, 0.6, 0.8]
cebra_modal_list = ['cebra_b', 'cebra_h', 'cebra_t']






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
    # ----------------- CEBRA MODEL TRAINING ----------------------- # 

    user_list = [1]
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
    
                        embedding_dir = f'./embeddings/{cebra_modal}/dataset{dataset}'
                        auxf.ensure_directory_exists(embedding_dir)

                        np.save(f"{embedding_dir}/{cebra_model_ID}.npy", embedding)
