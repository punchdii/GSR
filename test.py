import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from scipy import signal

data = pd.read_csv("moveandGsr.csv")

conductance = data['Shimmer_3B52_Accel_LN_Z_CAL']


def median_filter(data, window_size):
    filtered_data = np.zeros(len(data))
    for i in range(len(data)):
        start = max(0, i - window_size//2)
        end = min(len(data), i + window_size//2 + 1)
        window = data[start:end]
        filtered_data[i] = np.median(window)
    return filtered_data


data = median_filter(conductance, 3)
# Plot original and filtered signals



plt.plot(filtered_signal)
plt.show()


# conductance = data['conductance']
# time = data['Seconds']


