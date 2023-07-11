from numpy import linspace, log, sin, pi
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad,trapezoid
import numpy as np


def median_filter(data, window_size):
    filtered_data = np.zeros(len(data))
    for i in range(len(data)):
        start = max(0, i - window_size//2)
        end = min(len(data), i + window_size//2 + 1)
        window = data[start:end]
        filtered_data[i] = np.median(window)
    return filtered_data




data = pd.read_csv("accy50hz.csv")

ydata = data['Shimmer_3B52_Accel_LN_Y_CAL']
for i in range(len(ydata)):
    ydata[i] = ydata[i]-2

begin = 88
long = 62
ydata  = ydata[begin:begin+long]
ydata = median_filter(ydata,9)
xdata = np.linspace(0,0.02*long,long)

plt.subplot(221)
plt.title("Accelaration")
plt.xlabel('Time (s)')
plt.ylabel('m/s^2')
plt.plot(xdata,ydata,label = "Accelaration")
plt.legend()



newx = []
newy = []
start = 0
for i in range(len(ydata)):
    integ = trapezoid(ydata[start:i],xdata[start:i])
    #only integrate the negative values 
    #if ydata[i]>0 and xdata[i]<3:
      #  start = i+1
    newy.append(integ)

plt.subplot(222)
plt.title("velocity")
plt.xlabel('Time (s)')
plt.ylabel('m/s')
plt.plot(xdata,newy,label = "Velocity")
plt.legend()


dx = []
dy = []
start = 0
for i in range(len(newy)):
    integ = trapezoid(newy[start:i],xdata[start:i])
    #only integrate the negative values 
    #if ydata[i]>0 and xdata[i]<3:
      #  start = i+1
    dy.append(integ)


plt.subplot(223)
plt.title("Displacement")
plt.xlabel('Time (s)')
plt.ylabel('m')
plt.plot(xdata,dy,label = "displacement")
plt.legend()




coorx = round(xdata[-1],2)
coory = round(dy[-1],2)

label = f"({coorx},{coory})"
plt.annotate(label, # this is the text
                (coorx,coory), # these are the coordinates to position the label
                textcoords="offset points", # how to position the text
                xytext=(0,0), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center



plt.tight_layout(pad=2.0)
plt.legend()
plt.suptitle("Accelaration to Displacement")
plt.show()

final_result = dy[-1]
print("the integration for velocity which should be the displacement is",final_result)
