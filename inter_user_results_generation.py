# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import aux_functions as auxf
import cebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

params = {"font.family" : "serif"}
plt.rcParams.update(params)

np.random.seed(42)
torch.manual_seed(42)


class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

def calcRsq(user, dimred_type, dataset):
    y_hat = auxf.getGRU_pred(user=user, dimred_type=dimred_type, dataset=dataset)
    y = auxf.getProcessedData(user=user, dataset=dataset, mode='glove')

    if y.shape != y_hat.shape:
        y_hat = y_hat.T

    r2_list = [r2_score(y_true=y, y_pred=y_hat, multioutput='uniform_average')]
    for i in range(5):
        r2_list.append(r2_score(y_true=y[:, i], y_pred=y_hat[:, i]))

    return r2_list

def calcRsq_test(user_i, user_j, dimred_type):
    dataset = 3
    y_hat = auxf.getGRU_pred_test_ij(user_i, user_j, dimred_type=dimred_type)
    y = auxf.getProcessedData(user=user_j, dataset=dataset, mode='glove')

    if y.shape != y_hat.shape:
        y_hat = y_hat.T

    r2_list = [r2_score(y_true=y, y_pred=y_hat, multioutput='uniform_average')]
    for i in range(5):
        r2_list.append(r2_score(y_true=y[:, i], y_pred=y_hat[:, i]))

    return r2_list

# Add function for cross-user evaluation
def cross_user_evaluation(cebra_model, reg_GRU, user_list, user_i, dataset=3):
    r2_matrix = []
    dimred_type = "cebra_b"
    for user_j in user_list:
        # Get EMG data for user_j
        emg_test = np.load(f"./processed_data/user{user_j}/emg/dataset{dataset}/emg_{user_j}_{dataset}.npy")

        # Encode data using CEBRA model trained on user_i
        emg_embedding_test = cebra_model.transform(emg_test)

        # Decode using GRU model trained on user_i
        emg_test_tensor = torch.tensor(emg_embedding_test, dtype=torch.float32)
        reg_GRU.eval()
        with torch.no_grad():
            predictions_test = reg_GRU(emg_test_tensor.unsqueeze(1))

        # Save predictions
        pred_path = f"./GRU_pred_test/{dimred_type}/trained_user{user_i}"
        auxf.ensure_directory_exists(pred_path)
        np.save(f"{pred_path}/tr_user{user_i}_test_user{user_j}.npy", predictions_test.numpy())

        # Calculate R² scores for user_j
        r2_list = calcRsq_test(user_i=user_i, user_j = user_j, dimred_type="cebra_b")
        r2_matrix.append([user_i, user_j] + r2_list)

    return r2_matrix


def trainBestCEBRA(user: int):

    cebra_best_df = pd.read_csv('results_df/best_validation/r2_cebra_hyperp_best.csv')

    # for cebra_modal in cebra_modal_list:

    # hardcode: 
    cebra_modal = 'cebra_b'

    cebra_df_filtered = cebra_best_df[cebra_best_df['cebra_modal'] == cebra_modal]

    model_arch = cebra_df_filtered['model_arch'].values[0]
    min_temp_val = cebra_df_filtered['min_temp_val'].values[0]
    time_offset_val = cebra_df_filtered['time_offset_val'].values[0]

    dimred_ID = f"{model_arch}_{min_temp_val}_{time_offset_val}"
    cebra_modal = 'cebra_b'

    # now join dataset1 and 2 to form trainval and train with the best hyperparams
    
    dataset = 1

    emg_d1 = np.load(f"./processed_data/user{user}/emg/dataset{dataset}/emg_{user}_{dataset}.npy")
    glove_d1 = np.load(f"./processed_data/user{user}/glove/dataset{dataset}/glove_{user}_{dataset}.npy")

    dataset = 2

    emg_d2 = np.load(f"./processed_data/user{user}/emg/dataset{dataset}/emg_{user}_{dataset}.npy")
    glove_d2 = np.load(f"./processed_data/user{user}/glove/dataset{dataset}/glove_{user}_{dataset}.npy")

    emg_trainval = np.vstack((emg_d1, emg_d2))
    glove_trainval = np.vstack((glove_d1, glove_d2))


    # if cebra_modal == 'cebra_b':
    #     condit = 'time_delta'
    #     hybrid_bool = False

    # elif cebra_modal == 'cebra_h':
    #     condit = 'time_delta'
    #     hybrid_bool = True

    # elif cebra_modal == 'cebra_t':
    #     condit = 'time'
    #     hybrid_bool = False

    condit = 'time_delta'
    hybrid_bool = False
    cebra_modAl = 'cebra_b'



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
                        temperature_mode = 'auto',
                        min_temperature = min_temp_val, 
                        hybrid = hybrid_bool)


    if cebra_modAl == 'cebra_b' or cebra_modAl == 'cebra_h':
        cebra_model.fit(emg_trainval, glove_trainval)

    emg_embedding_trainval = cebra_model.transform(emg_trainval)


    cebra_model_dir = f'./cebra_models_test/user{user}'
    auxf.ensure_directory_exists(cebra_model_dir)
    cebra_model.save(f"{cebra_model_dir}/{dimred_ID}.pt")



    return cebra_model


def trainGRU(user: int, cebra_model_, dimred_type: str):

    dataset = 1

    emg_d1 = np.load(f"./processed_data/user{user}/emg/dataset{dataset}/emg_{user}_{dataset}.npy")
    glove_d1 = np.load(f"./processed_data/user{user}/glove/dataset{dataset}/glove_{user}_{dataset}.npy")

    dataset = 2

    emg_d2 = np.load(f"./processed_data/user{user}/emg/dataset{dataset}/emg_{user}_{dataset}.npy")
    glove_d2 = np.load(f"./processed_data/user{user}/glove/dataset{dataset}/glove_{user}_{dataset}.npy")

    emg_trainval = np.vstack((emg_d1, emg_d2))
    glove_trainval = np.vstack((glove_d1, glove_d2))

    emg_embedding_trainval = cebra_model_.transform(emg_trainval)

    emg_embedding_trainval_tensor = torch.tensor(emg_embedding_trainval, dtype=torch.float32)
    glove_trainval_tensor = torch.tensor(glove_trainval, dtype=torch.float32)

    # # -- test data --- #

    # dataset = 3

    # emg_test = np.load(f"./processed_data/user{user}/emg/dataset{dataset}/emg_{user}_{dataset}.npy")
    # emg_embedding_test = cebra_model.transform(emg_test)

    # embedding_dir = f'./embeddings_test/user{user}/{dimred_type}/{dataset}'
    # auxf.ensure_directory_exists(embedding_dir)

    # np.save(f"{embedding_dir}/{dimred_type}.npy", emg_embedding_test)

    # ---- GRU --- #

    input_size = emg_embedding_trainval.shape[1]  # assuming the embedding dimension
    hidden_size = 130
    output_size = glove_trainval.shape[1] 
    num_layers = 2


    # train a GRU on trainval

    reg_GRU = GRURegressor(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(reg_GRU.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        reg_GRU.train()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (training)
        outputs_train = reg_GRU(emg_embedding_trainval_tensor.unsqueeze(1))  # Add sequence dimension
        train_loss = criterion(outputs_train, glove_trainval_tensor)

        # Backward pass and optimization
        train_loss.backward()
        optimizer.step()

    reg_GRU.eval()

    return reg_GRU




iters_CEBRA = 22000
num_epochs = 600

# Define the model architecture
input_size = 3  # This should be the dimension of your embeddings
hidden_size = 130  # As used in your GRU model
output_size = 5  # Assuming you are predicting 5 values (like DoA1, DoA2, etc.)
num_layers = 2  # As per your original model architecture


# Cross-user testing across all users
user_list = np.arange(1, 13)  # List of users
r2_cross_results = []

best_cebra_model_ID = "offset36-model_0.2_10"

dimred_type = 'cebra_b'


for user_i in user_list:
    # Train CEBRA and GRU on user_i as per your existing code

    # cebra_model = cebra.CEBRA.load(f'./cebra_models/user{user_i}/cebra_b/{best_cebra_model_ID}.pt')

    cebra_model = trainBestCEBRA(user = user_i)
    reg_GRU = trainGRU(user = user_i, cebra_model_ = cebra_model, dimred_type='cebra_b')
    
    # # Initialize the model with the same architecture as when you saved it
    # reg_GRU = GRURegressor(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)

    # # Load the saved weights
    # checkpoint_path = f'trained_GRU/user{user_i}/cebra_b/{best_cebra_model_ID}.pt'  # Provide the path to your saved .pt file
    # reg_GRU.load_state_dict(torch.load(checkpoint_path))

    # # Set the model to evaluation mode if you are evaluating
    # reg_GRU.eval()

    # Now evaluate cross-user performance
    r2_matrix = cross_user_evaluation(cebra_model, reg_GRU, user_list, user_i, dataset=3)
    r2_cross_results.extend(r2_matrix)

    print(r2_cross_results)

# Save R² matrix
r2_cross_df = pd.DataFrame(r2_cross_results, columns=['user_i', 'user_j', 'mvr2', 'DoA1', 'DoA2', 'DoA3', 'DoA4', 'DoA5'])
r2_cross_path = f'./results_df/{dimred_type}'
auxf.ensure_directory_exists(r2_cross_path)
r2_cross_df.to_csv(f'{r2_cross_path}/cross_user_r2_matrix.csv', index=False)


# ---------- PLOTTING ----------- #

# Load the cross-user R² matrix from the CSV file
r2_cross_df = pd.read_csv(f'{r2_cross_path}/cross_user_r2_matrix.csv')

# Pivot the DataFrame to have 'user_j' as columns and 'user_i' as rows, with R² as values
# Assuming we want to plot the multivariate R² score ('mvr2')
r2_pivot = r2_cross_df.pivot(index='user_i', columns='user_j', values='mvr2')

# Create the heatmap
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(r2_pivot, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", cbar_kws={'label': 'R² Score'})

# Set the labels and title
plt.xlabel('Tested User')
plt.ylabel('Trained User')
plt.title('Cross-User R² Heatmap (Train vs Test)')

# Show the plot
plt.show()

# -- #


avg_r2_per_user = r2_cross_df.groupby('user_i')['mvr2'].mean()

figpath = './results_fig/cebra_b'
auxf.ensure_directory_exists(figpath)

fig = 1
# Plot the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_r2_per_user.index, y=avg_r2_per_user.values, palette="viridis")
plt.xlabel('Trained User')
plt.ylabel('Average R²')
plt.title('Average R² for Each Trained User (Across All Tested Users)')
plt.savefig(f"./{figpath}/fig{fig}.png")

# ---- #

fig = 2
# Melt the DataFrame for all DoAs
r2_long = r2_cross_df.melt(id_vars=['user_i', 'user_j'], value_vars=['DoA1', 'DoA2', 'DoA3', 'DoA4', 'DoA5'],
                           var_name='DoA', value_name='R²')

# Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='DoA', y='R²', data=r2_long, palette='Set3')
plt.xlabel('Degree of Agreement (DoA)')
plt.ylabel('R² Score')
plt.title('R² Distribution for Each Degree of Agreement (DoA) Across Users')
plt.savefig(f"./{figpath}/fig{fig}.png")

# ----

fig = 3
plt.figure(figsize=(10, 8))
sns.clustermap(r2_pivot, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", metric="euclidean", method="average")
plt.title('Clustered Heatmap of Cross-User R² Scores')
plt.savefig(f"./{figpath}/fig{fig}.png")
