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

model_arch = 'offset5-model'
time_offset_val = 2

iters = 10

glove = np.load("./processed_data/glove/dataset3/glove_1_3.npy")
emg = np.load("./processed_data/emg/dataset3/emg_1_3.npy")

cebra_modal = 'cebra_b'

time_offset_dict = {
    "offset5-model" : [1, 2, 4], 
    "offset10-model" : [2, 4, 8],
    "offset36-model" : [2, 10, 20, 32]
}


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

                cebra_model_dir = f'./cebra_models/{cebra_modal}'
                auxf.ensure_directory_exists(cebra_model_dir)
                cebra_model_ID = f"{model_arch}_{min_temp_val}_{time_offset_val}"
                cebra_model.save(f"{cebra_model_dir}/{cebra_model_ID}.pt")