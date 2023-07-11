from numpy import linspace, log, sin, pi
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np


def median_filter(data, window_size):
    filtered_data = np.zeros(len(data))
    for i in range(len(data)):
        start = max(0, i - window_size//2)
        end = min(len(data), i + window_size//2 + 1)
        window = data[start:end]
        filtered_data[i] = np.median(window)
    return filtered_data



s = 0
e = 1600
r = 0.90
data = pd.read_csv("new_door_test.csv")

acdata = data['Shimmer_3B52_Accel_LN_Z_CAL']
acdata = acdata[s:e]
for i in range(len(acdata)):
    acdata[i] = acdata[i]-2



acdata  = acdata[s:e]
acdata = median_filter(acdata,1)
xdata = np.linspace(s,e/404.54,e)

gydata = data['Shimmer_3B52_Gyro_Y_CAL']
gydata = gydata[s:e]
gydata = median_filter(gydata,1)
for i in range(len(gydata)):
    gydata[i] = -gydata[i]+1
    gydata[i] = (gydata[i]*pi)/180


#graphing tangential accleration graph
a2 = []
for i in range(len(xdata)):
    if i<1599:
        a2.append(r*((gydata[i+1]-gydata[i])/(xdata[i+1]-xdata[i])))
        
    #dydx = np.gradient(y, dx)
a2 = median_filter(a2,200)

plt.subplot(224)
plt.title("tangential accleration")
plt.plot(xdata[0:1599],a2, label = "tengential")
plt.xlabel('Time (s)')
plt.ylabel("Acceleration (m/s^2)")
plt.legend()

#graphing centropical accleration graph
""" a1 = []
for i in range(len(xdata)):
    acc = r*gydata[i]**2
    a1.append(acc)
 """
""" plt.subplot(221)
plt.title("centropical accleration")
plt.plot(xdata,a1, label = "centropecal")
plt.xlabel('Time (s)')
plt.ylabel("Acceleration (m/s^2)")
plt.legend()
 """
#accelerometer original graph
plt.subplot(222)
plt.plot(xdata,acdata,label = "z-accelaration")
plt.title('z-acceleration')
plt.xlabel('Time (s)')
plt.ylabel("Acceleration (m/s^2)")
plt.legend()


#gyroscope orginal graph
plt.subplot(223)
plt.plot(xdata,gydata,label = "y-gyroscope data")
plt.xlabel('Time (s)')
plt.ylabel("Velocity (m/s)")
plt.legend()

plt.tight_layout(pad = 2.0)
plt.show()