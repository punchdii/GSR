from numpy import linspace, log, sin, pi
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad, trapezoid
import numpy as np

def median_filter(data, window_size):
    filtered_data = np.zeros(len(data))
    for i in range(len(data)):
        start = max(0, i - window_size//2)
        end = min(len(data), i + window_size//2 + 1)
        window = data[start:end]
        filtered_data[i] = np.median(window)
    return filtered_data



long = 1100
data = pd.read_csv("gyro_precise.csv")

ydata = data['Shimmer_3B52_Gyro_Y_CAL']
ydata = ydata[200:200+long]
ydata = median_filter(ydata,101)
xdata = list(np.linspace(0,1/203*long,long))



newx = []
newy = []
start = 0
for i in range(len(ydata)):
    integ = trapezoid(ydata[start:i],xdata[start:i])
    #only integrate the negative values 
    if ydata[i]>0 and xdata[i]<3:
        start = i+1
    newy.append(integ)
    #if the next item is larger than the previous one, ignore it and replace it with the previous item
    if i>1 and newy[i] >newy[i-1]:
        newy[i] = newy[i-1]

plt.subplot(122)
plt.title("Rotation")
plt.plot(xdata,newy,label = "angle")
plt.xlabel('Time (s)')
plt.ylabel('Degree')

coorx = round(xdata[-1],2)
coory = round(newy[-1],2)

label = f"({coorx},{coory})"
plt.annotate(label, # this is the text
                (coorx,coory), # these are the coordinates to position the label
                textcoords="offset points", # how to position the text
                xytext=(0,0), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center



plt.subplot(121)
plt.title("Angular Velocity")
plt.plot(xdata,ydata,label = "original")
plt.xlabel('Time (s)')
plt.ylabel('Degree/s')
plt.legend()
plt.show()