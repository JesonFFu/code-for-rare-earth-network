from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.metrics import mean_squared_error

def nrmse_via_mse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denominator = np.std(y_true)
    return rmse / denominator

# --- Data loading and initial preprocessing (moved outside the loop) ---
x1 = np.loadtxt('new2000.csv')
x1 = x1.reshape(-1,)
L = len(x1)
maxx1 = np.max(x1)
minx1 = np.min(x1)
x1 = (x1 - minx1) / (maxx1 - minx1) * 7 + 2

data = sio.loadmat('MG maichongxitu.mat')
data1 = np.array(data['y5'])
data2 = data1[0, 2000:-1]
# Process data3_processed_np once
data3_processed_np = data2[:].reshape(-1, 10)[:, 9] # This should be 1D
# --- End of moved block ---

l_nrmse = []
k = 50 # mask length

# Ensure data3_processed_np is long enough for L*k for the temp1 creation
if len(data3_processed_np) < L * k:
    raise ValueError(f"data3_processed_np is too short ({len(data3_processed_np)}) for L*k ({L*k})")

for l in range(0, 51):
    # Optimized temp1 generation (vectorized)
    # This assumes data3_processed_np is a 1D array of sufficient length
    temp1 = data3_processed_np[:L*k].reshape(L, k)

    m = 0 # future-current interval i

    # Optimized temp2 generation (vectorized concatenation)
    if l > 0:
        # Create a list of shifted temp1 arrays for concatenation
        shifted_temps = [temp1[i:L-l+i, :] for i in range(l)]
        # Concatenate them horizontally
        temp2 = np.concatenate(shifted_temps, axis=1)
    else:
        temp2 = temp1

    Samples = temp2
    Labels = x1[l+1+m:]

    Samples1 = Samples[int(L*0.1):int(L*0.7), :]
    Labels1 = Labels[int(L*0.1):int(L*0.7)]


    clf = MLPRegressor(alpha=0.1, hidden_layer_sizes = (), max_iter = 100000, tol=1e-7,
                        activation = 'identity', verbose = False, learning_rate = 'constant', learning_rate_init = 0.0001, random_state=35)
    clf.fit(Samples1,Labels1) # 'a' variable was not used, removed assignment

    # Prediction on test set
    Samples_test = Samples[int(L*0.7):int(L*0.9), :]
    Labels_test = Labels[int(L*0.7):int(L*0.9)]

    Y1 = clf.predict(Samples_test)

    # Calculate NRMSE
    current_nrmse = nrmse_via_mse(Labels_test, Y1)
    print(f"l={l}, NRMSE: {current_nrmse}")
    l_nrmse.append(current_nrmse)

iii=range(0,51) # Changed from 1 to 0 to match loop range
plt.plot(iii,l_nrmse,"r",markersize = 2)
plt.xlabel("l value")
plt.ylabel("NRMSE")
plt.title("NRMSE vs. l (Optimized NumPy)")
plt.show()
b=1