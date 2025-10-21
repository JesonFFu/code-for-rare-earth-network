from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
from sklearn.metrics import mean_squared_error

def nrmse_via_mse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denominator = np.std(y_true)
    # Handle the case where std is zero to avoid division by zero
    if denominator == 0:
        return np.inf if rmse > 0 else 0
    return rmse / denominator

def run_mlp_experiment_f(random_seed, output_base_dir='E:\\efg\\f'):


    print(f"\n--- Running experiment with random_state = {random_seed} ---")

    # --- Data loading and initial preprocessing (outside m loop for efficiency) ---
    # Ensure these paths are correct for your system
    x1_path = 'new2000.csv'  # Assuming it's in the same directory as the script
    mat_data_path = 'MG maichongxitu.mat' # Assuming it's in the same directory

    if not os.path.exists(x1_path):
        raise FileNotFoundError(f"File not found: {x1_path}")
    if not os.path.exists(mat_data_path):
        raise FileNotFoundError(f"File not found: {mat_data_path}")

    x1 = np.loadtxt(x1_path)
    x1 = x1.reshape(-1,)
    L = len(x1)
    maxx1 = np.max(x1)
    minx1 = np.min(x1)
    
    # Check for division by zero if maxx1 == minx1
    if (maxx1 - minx1) == 0:
        x1_normalized = np.zeros_like(x1)
    else:
        x1_normalized = (x1 - minx1) / (maxx1 - minx1) * 7 + 2

    data = sio.loadmat(mat_data_path)
    data1 = np.array(data['y5'])
    data2 = data1[0, 2000:-1]
    
    # Process data3_raw once outside the m loop
    data3_raw = data2[:] # Keep a reference to the raw data segment
    data3_processed_for_temp1 = data3_raw.reshape(-1, 10)[:, 9] # This is a 1D array

    k = 50
    l = 20 # fixed 'l' value as per your code
    
    m_nrmse = []

    # Ensure data3_processed_for_temp1 is long enough for temp1 creation
    required_data3_len_for_temp1 = L * k
    if len(data3_processed_for_temp1) < required_data3_len_for_temp1:
        print(f"Warning: data3_processed_for_temp1 is too short ({len(data3_processed_for_temp1)}) "
              f"for L*k ({required_data3_len_for_temp1}). Padding with zeros.")
        data3_processed_for_temp1 = np.pad(data3_processed_for_temp1, 
                                            (0, required_data3_len_for_temp1 - len(data3_processed_for_temp1)), 
                                            'constant', constant_values=0)


    # Vectorized temp1 generation
    # Reshape data3_processed_for_temp1 into (L, k) directly
    temp1 = data3_processed_for_temp1[:L*k].reshape(L, k)


    for m in range(0, 51):

        if l > 0:

            shifted_arrays = [temp1[j:L-l+j, :] for j in range(l)]
            

            expected_rows = L - l
            
            # Check if any shifted array would be empty or have fewer rows than expected
            valid_shifted_arrays = []
            for j in range(l):
                start_idx = j
                end_idx = j + expected_rows
                
                # Ensure the slice is within temp1 bounds
                if start_idx < L and end_idx <= L:
                    valid_shifted_arrays.append(temp1[start_idx:end_idx, :])
                elif start_idx < L: # Partial slice
                    valid_shifted_arrays.append(temp1[start_idx:L, :])
                else: # empty
                    valid_shifted_arrays.append(np.empty((0,k)))

            if len(valid_shifted_arrays) == 0 or any(arr.shape[0] == 0 for arr in valid_shifted_arrays):
                 temp2 = np.empty((0, k*l)) # Handle cases where no valid shifts are possible
            else:
                 # Ensure all valid_shifted_arrays have the same number of rows for concatenation
                 min_rows = min(arr.shape[0] for arr in valid_shifted_arrays)
                 if min_rows > 0:
                    temp2 = np.concatenate([arr[:min_rows, :] for arr in valid_shifted_arrays], axis=1)
                 else:
                    temp2 = np.empty((0, k*l))

        else: # l=0
            temp2 = temp1 # This case is not hit with l=20 fixed

        Samples = temp2


        effective_num_samples = Samples.shape[0]

        # Ensure x1_normalized has enough elements for Labels
        labels_start_idx = l + m
        labels_end_idx = l + m + effective_num_samples
        
        if labels_end_idx > len(x1_normalized):
            print(f"Warning: x1_normalized is too short ({len(x1_normalized)}) "
                  f"for Labels required up to index {labels_end_idx}. Adjusting Samples length.")
            effective_num_samples = len(x1_normalized) - labels_start_idx
            if effective_num_samples < 0:
                effective_num_samples = 0
            Samples = Samples[:effective_num_samples, :]
            
        Labels = x1_normalized[labels_start_idx : labels_start_idx + effective_num_samples]
        
        # Final check for consistency
        min_rows_sl = min(Samples.shape[0], Labels.shape[0])
        Samples = Samples[:min_rows_sl, :]
        Labels = Labels[:min_rows_sl]


        if Samples.shape[0] == 0:
            print(f"Skipping m={m} due to empty samples after alignment.")
            m_nrmse.append(np.nan)
            continue

        # Split data for training and testing
        total_rows = Samples.shape[0]
        train_start_idx = int(total_rows * 0.1)
        train_end_idx = int(total_rows * 0.7)
        test_start_idx = int(total_rows * 0.7)
        test_end_idx = int(total_rows * 0.9)

        Samples1 = Samples[train_start_idx:train_end_idx, :]
        Labels1 = Labels[train_start_idx:train_end_idx]

        Samples_test = Samples[test_start_idx:test_end_idx, :]
        Labels_test = Labels[test_start_idx:test_end_idx]

        # Ensure there's data in the training and testing sets
        if Samples1.shape[0] == 0 or Samples_test.shape[0] == 0:
            print(f"Skipping m={m} due to insufficient data for train/test split.")
            m_nrmse.append(np.nan)
            continue
            

        clf = MLPRegressor(alpha=0.1, hidden_layer_sizes=(), max_iter=100000, tol=1e-7,
                           activation='identity', verbose=False, learning_rate='constant',
                           learning_rate_init=0.0001, random_state=random_seed)
        clf.fit(Samples1, Labels1)

        # Prediction on test set (vectorized)
        Y1_pred = clf.predict(Samples_test)

        # Calculate NRMSE
        current_nrmse = nrmse_via_mse(Labels_test, Y1_pred)
        print(f"m={m}, NRMSE: {current_nrmse}")
        m_nrmse.append(current_nrmse)

    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Save NRMSE results to a .npy file
    output_filename = os.path.join(output_base_dir, f'm_nrmse_random_state_{random_seed}.npy')
    np.save(output_filename, np.array(m_nrmse))
    print(f"Saved NRMSE results for random_state={random_seed} to {output_filename}")
    
    # Plotting for the current random_state
    plt.figure(figsize=(10, 6))
    iii = range(0, 51) # Changed from 1 to 0 to match loop range
    plt.plot(iii, m_nrmse, "r", markersize=2)
    plt.xlabel("m value")
    plt.ylabel("NRMSE")
    plt.title(f"NRMSE vs. m (random_state={random_seed})")
    plt.grid(True)
    plt.savefig(os.path.join(output_base_dir, f'nrmse_plot_random_state_{random_seed}.png'))
    plt.close() # Close the plot to prevent it from displaying immediately for each run
    
    print(f"Plot saved for random_state={random_seed}.")
    
    return m_nrmse

# --- Main loop to run experiments for different random states ---
random_states_to_test = range(42, 43) # From 30 to 42
all_nrmse_results_f = {}

for rs in random_states_to_test:
    try:
        current_m_nrmse = run_mlp_experiment_f(rs)
        all_nrmse_results_f[rs] = current_m_nrmse
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
        break # Stop if essential files are missing
    except Exception as e:
        print(f"An unexpected error occurred for random_state={rs}: {e}")
        all_nrmse_results_f[rs] = [np.nan] * 51 # Store NaNs for failed runs

# --- Optional: Load and plot all results together ---
print("\n--- All experiments finished. ---")
print("Loading and plotting all results together...")

output_base_dir = 'E:\\efg\\f'
plt.figure(figsize=(12, 8))
for rs in random_states_to_test:
    filename = os.path.join(output_base_dir, f'm_nrmse_random_state_{rs}.npy')
    if os.path.exists(filename):
        loaded_nrmse = np.load(filename)
        plt.plot(range(0, 51), loaded_nrmse, label=f'random_state={rs}')
    else:
        print(f"Warning: File not found for random_state={rs}: {filename}")

plt.xlabel("m value")
plt.ylabel("NRMSE")
plt.title("NRMSE vs. m for different random_states")
plt.legend()
plt.grid(True)
plt.show()
b=1