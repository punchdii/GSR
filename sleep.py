import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from scipy import signal
from statsmodels.tsa.seasonal import STL



data = pd.read_csv("sleep2.csv")

conductance = data['Shimmer_3B52_GSR_Skin_Conductance_CAL']

stl = STL(conductance)
result = stl.fit()
seasonal,trend,resid = result.seasonal,result.trend,result.resid


def median_filter(data, window_size):
    filtered_data = np.zeros(len(data))
    for i in range(len(data)):
        start = max(0, i - window_size//2)
        end = min(len(data), i + window_size//2 + 1)
        window = data[start:end]
        filtered_data[i] = np.median(window)
    return filtered_data



filtered_data = median_filter(conductance, 901)
# Plot original and filtered signals
data_detrend = signal.detrend(filtered_data,type='linear',overwrite_data=False)

for i in range(len(data_detrend)):
    data_detrend[i] = data_detrend[i]+0.8


#plt.plot(conductance,label = "unfiltered")
#plt.plot(filtered_data,label = "filtered data")
plt.plot(data_detrend,label  = "detrended data")
plt.plot(seasonal,label=  "seasonal")
plt.plot(trend,label = "trend")
plt.legend()
plt.show()

