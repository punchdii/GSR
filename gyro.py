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



data = pd.read_csv("gyro.csv")

ydata = data['Shimmer_3B52_Gyro_Z_CAL']
ydata = ydata[33:43]
ydata = median_filter(ydata,3)
xdata = list(np.linspace(0,0.666666666,10))



def func(x,a,b,c):
    return a*x*x+b*x+c

constants = curve_fit(func,xdata,ydata)
a_fit= constants[0][0]
b_fit= constants[0][1]
c_fit= constants[0][2]


a = a_fit
b = b_fit
c = c_fit

def f(x):
    return a*x*x+b*x+c

x= []
y = []
for i in xdata:
    x.append(i)
    y.append(f(i))
    
plt.subplot(121)
plt.xlabel('Time (s)')
plt.ylabel('Deg/s')
plt.plot(x,y,label = "approximation")
plt.plot(xdata,ydata,label = "orginal")
plt.legend()


dx = []
dy = []
for t in xdata:
    integ = quad(f,0,t)
    dx.append(t)
    dy.append(integ[0])
    if t == xdata[-1]:
        coorx = round(t,2)
        coory = round(integ[0],2)



plt.subplot(122)
plt.plot(dx,dy,'o-',label = "angle(integration)")

print(a_fit,b_fit,c_fit)



plt.xlabel('Time (s)')
plt.ylabel('Degree')
label = f"({coorx},{coory})"
plt.annotate(label, # this is the text
                (coorx,coory), # these are the coordinates to position the label
                textcoords="offset points", # how to position the text
                xytext=(0,0), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center



plt.legend()
plt.show()