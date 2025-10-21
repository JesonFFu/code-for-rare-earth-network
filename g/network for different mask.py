from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
from sklearn.metrics import mean_squared_error
import warnings
import time

def nrmse_via_mse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denominator = np.std(y_true)
    # Handle the case where std is zero to avoid division by zero
    if denominator == 0:
        return np.inf if rmse > 0 else 0
    return rmse / denominator

def run_mlp_experiment_g(random_seed, output_base_dir='E:\\efg图\\g'):

    print(f"\n--- Running experiment with random_state = {random_seed} ---")
    start_time_exp = time.time()

    # --- Data loading and initial preprocessing (fixed for x1) ---
    x1_path = 'new2000.csv' # Assuming this is in the current working directory or accessible
    
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    if not os.path.exists(x1_path):
        # Fallback if not in cwd, try to find it in output_base_dir
        if os.path.exists(os.path.join(output_base_dir, x1_path)):
            x1_path = os.path.join(output_base_dir, x1_path)
        else:
            raise FileNotFoundError(f"File not found: {x1_path}")

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

    k_nrmse = []

    # Outer loop for 'k' (from 1 to 50)
    for k_val in range(1, 51):
        print(f"  - Processing k = {k_val}")
        mat_data_path = os.path.join(output_base_dir, f'MG maichongxitu_{k_val}.mat')
        

        if not os.path.exists(mat_data_path):
            # Attempt to find it in the current working directory if not in output_base_dir
            if os.path.exists(f'MG maichongxitu_{k_val}.mat'):
                mat_data_path = f'MG maichongxitu_{k_val}.mat'
            else:
                warnings.warn(f"File not found: {mat_data_path}. Skipping k={k_val}.")
                k_nrmse.append(np.nan)
                continue # Skip to next k_val

        try:
            data = sio.loadmat(mat_data_path)
            data1 = np.array(data['y5'])
            data2 = data1[0, 2000:-1]
            data3_raw = data2[:] # Use a more descriptive name
        except Exception as e:
            warnings.warn(f"Error loading or processing {mat_data_path}: {e}. Skipping k={k_val}.")
            k_nrmse.append(np.nan)
            continue

        # --- Optimized temp1 generation ---
        # data3_raw is expected to be 1D. If it's not, flatten it.
        if data3_raw.ndim > 1:
            data3_raw_1d = data3_raw.flatten()
        else:
            data3_raw_1d = data3_raw
        
        # Ensure data3_raw_1d is long enough and divisible by 10 for reshaping
        processed_data3_len = (len(data3_raw_1d) // 10) * 10
        if processed_data3_len == 0:
            warnings.warn(f"data3_raw is too short for k={k_val}. Skipping.")
            k_nrmse.append(np.nan)
            continue
        
        data3_processed_for_temp1 = data3_raw_1d[:processed_data3_len].reshape(-1, 10)[:, 9]
        
        # Now, create temp1 efficiently
        # We need L rows, each with k_val elements.
        # data3_processed_for_temp1 must have at least L * k_val elements.
        
        required_len_for_temp1 = L * k_val
        if len(data3_processed_for_temp1) < required_len_for_temp1:
            warnings.warn(f"data3_processed_for_temp1 too short for L={L}, k={k_val}. Padding with zeros.")
            data3_processed_for_temp1 = np.pad(data3_processed_for_temp1, 
                                                (0, required_len_for_temp1 - len(data3_processed_for_temp1)), 
                                                'constant', constant_values=0)
        
        # Reshape directly for temp1
        temp1 = data3_processed_for_temp1[:required_len_for_temp1].reshape(L, k_val)
        
        l = 20
        m = 0 # m is fixed at 0 as per your second code structure
 
        rows_for_temp2 = L - l
        if rows_for_temp2 <= 0:
            warnings.warn(f"L ({L}) is not large enough for l ({l}). Samples will be empty. Skipping k={k_val}.")
            k_nrmse.append(np.nan)
            continue

        temp2_parts = []
        for j in range(l):
            temp2_parts.append(temp1[j : j + rows_for_temp2, :])
        
        Samples = np.concatenate(temp2_parts, axis=1) # This is the main Samples matrix

        # Labels are x1_normalized[l+1+m:]
        # The length of Samples is rows_for_temp2.
        # So Labels should also have rows_for_temp2 elements, starting from index l+1+m.
        
        labels_start_idx = l + 1 + m
        effective_num_samples = Samples.shape[0]
        labels_end_idx = labels_start_idx + effective_num_samples
        
        if labels_end_idx > len(x1_normalized):
            new_effective_num_samples = len(x1_normalized) - labels_start_idx
            if new_effective_num_samples < 0:
                new_effective_num_samples = 0
            
            if new_effective_num_samples < effective_num_samples:
                warnings.warn(f"For k={k_val}, x1_normalized is too short. Reducing sample count from "
                              f"{effective_num_samples} to {new_effective_num_samples}.")
                effective_num_samples = new_effective_num_samples
        
        if effective_num_samples <= 0:
            warnings.warn(f"Skipping k={k_val} due to zero or negative effective samples after label alignment.")
            k_nrmse.append(np.nan)
            continue
        
        Samples = Samples[:effective_num_samples, :]
        Labels = x1_normalized[labels_start_idx : labels_start_idx + effective_num_samples]

        # Double-check final alignment after trimming
        min_rows_sl = min(Samples.shape[0], Labels.shape[0])
        Samples = Samples[:min_rows_sl, :]
        Labels = Labels[:min_rows_sl]
        
        if Samples.shape[0] == 0:
            warnings.warn(f"Skipping k={k_val} due to empty samples after final alignment.")
            k_nrmse.append(np.nan)
            continue


        # Split data for training and testing
        total_rows = Samples.shape[0]
        train_start_idx = int(total_rows * 0.1) # Changed from 0 to 0.1 as in your previous version for consistency
        train_end_idx = int(total_rows * 0.7)
        test_start_idx = int(total_rows * 0.7)
        test_end_idx = int(total_rows * 0.9) # Test set is 0.7 to 0.9 of total data

        # Ensure split indices are valid
        if train_end_idx <= train_start_idx or test_end_idx <= test_start_idx or \
           test_start_idx >= total_rows or train_end_idx >= total_rows:
            warnings.warn(f"Insufficient data for train/test split for k={k_val}. Total rows: {total_rows}. Skipping.")
            k_nrmse.append(np.nan)
            continue
        
        Samples1 = Samples[train_start_idx:train_end_idx, :]
        Labels1 = Labels[train_start_idx:train_end_idx]

        Samples_test = Samples[test_start_idx:test_end_idx, :]
        Labels_test = Labels[test_start_idx:test_end_idx]

        if Samples1.shape[0] == 0 or Samples_test.shape[0] == 0:
            warnings.warn(f"Empty training or testing sets for k={k_val}. Train shape: {Samples1.shape}, Test shape: {Samples_test.shape}. Skipping.")
            k_nrmse.append(np.nan)
            continue

        clf = MLPRegressor(alpha=0.1, hidden_layer_sizes = (), max_iter = 100000, tol=1e-7,
                            activation = 'identity', verbose = False, learning_rate = 'constant',
                            learning_rate_init = 0.0001, random_state=random_seed) # Use the passed random_seed

        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Ignore convergence warnings
            clf.fit(Samples1, Labels1)

        # --- Optimized Prediction ---
        # Predict on the entire test set at once
        Y1_pred = clf.predict(Samples_test)

        # Calculate NRMSE using the actual test labels
        current_nrmse = nrmse_via_mse(Labels_test, Y1_pred)
        print(f"  - k={k_val}, NRMSE: {current_nrmse}")
        k_nrmse.append(current_nrmse)


    output_filename = os.path.join(output_base_dir, f'k_nrmse_random_state_{random_seed}.npy')
    np.save(output_filename, np.array(k_nrmse))
    print(f"Saved NRMSE results for random_state={random_seed} to {output_filename}")


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 51), k_nrmse, "r", markersize=2) # k_val from 1 to 50
    plt.xlabel("k value")
    plt.ylabel("NRMSE")
    plt.title(f"NRMSE vs. k (random_state={random_seed})")
    plt.grid(True)
    plot_filename = os.path.join(output_base_dir, f'nrmse_plot_random_state_{random_seed}.png')
    plt.savefig(plot_filename)
    plt.close() # Close the plot to prevent it from displaying immediately for each run
    print(f"Plot saved for random_state={random_seed} to {plot_filename}")

    end_time_exp = time.time()
    print(f"Experiment for random_state={random_seed} completed in {end_time_exp - start_time_exp:.2f} seconds.")
    return k_nrmse


random_states_to_test = range(30, 43) # Example: run for states 30, 31, ..., 42
all_k_nrmse_results = {}
output_dir = 'E:\\efg\\g' # Your specified output directory

print("\n--- Starting all experiments for different random states ---")
total_start_time_all_runs = time.time()

for rs in random_states_to_test:
    try:
        current_k_nrmse = run_mlp_experiment_g(rs, output_base_dir=output_dir)
        all_k_nrmse_results[rs] = current_k_nrmse
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}. Please ensure 'new2000.csv' and 'MG maichongxitu_XX.mat' files are accessible.")
        break # Stop if essential files are missing
    except Exception as e:
        print(f"An unexpected error occurred for random_state={rs}: {e}")
        all_k_nrmse_results[rs] = [np.nan] * 50 # Store NaNs for failed runs for all k values

total_end_time_all_runs = time.time()
print(f"\n--- All experiments finished in {total_end_time_all_runs - total_start_time_all_runs:.2f} seconds. ---")

# --- Optional: Load and plot all results together ---
print("\nLoading and plotting all results together...")

plt.figure(figsize=(12, 8))
for rs in random_states_to_test:
    filename = os.path.join(output_dir, f'k_nrmse_random_state_{rs}.npy')
    if os.path.exists(filename):
        loaded_nrmse = np.load(filename)
        # Filter out NaN values for plotting if any runs failed partially
        valid_k_nrmse = np.array([val for val in loaded_nrmse if not np.isnan(val)])
        valid_k_range = np.arange(1, len(valid_k_nrmse) + 1)
        if len(valid_k_nrmse) > 0:
            plt.plot(valid_k_range, valid_k_nrmse, label=f'random_state={rs}')
        else:
            print(f"Warning: No valid NRMSE values to plot for random_state={rs} from {filename}")
    else:
        print(f"Warning: NRMSE file not found for random_state={rs}: {filename}")

plt.xlabel("k value")
plt.ylabel("NRMSE")
plt.title("NRMSE vs. k for different random_states")
plt.legend()
plt.grid(True)
plt.show()
B=1 