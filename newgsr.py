from numpy import linspace, log, sin, pi
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad,trapezoid
import numpy as np
from scipy.signal import butter,filtfilt
from cvxEDA import*

def median_filter(data, window_size):
    filtered_data = np.zeros(len(data))
    for i in range(len(data)):
        start = max(0, i - window_size//2)
        end = min(len(data), i + window_size//2 + 1)
        window = data[start:end]
        filtered_data[i] = np.median(window)

    return filtered_data

def remove_close_values(numbers, numbersy,threshold):
    #numbers.sort()  # Sort the list in ascending order
    
    i = 0
    while i < len(numbers) - 1:
        if abs(numbers[i] - numbers[i+1]) < threshold:
            del numbers[i+1]
            del numbersy[i+1]

             # Remove the next number if it's close
        else:
            i += 1  # Move to the next number if it's not close
    
    return numbers, numbersy



data = pd.read_csv("newgsr.csv")

ydata = data['Shimmer_3B52_GSR_Skin_Conductance_CAL']
begin = 6000
long = 15000
#29581
ydata  = ydata[begin:begin+long]

#median filter with a window of 1 second
xdata = np.linspace(0,long/128,long)


small = min(ydata)
big = max(ydata)

#preprocessing
ydata = median_filter(ydata,129)
for i in range(len(ydata)):
    ydata[i] = (ydata[i]-small)/(big-small)

onsetx = []
onsety = []
offsetx = []
offsety = []
#onset and offset
[r, p, t, l, d, e, obj] = cvxEDA(ydata, 0.0078125, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2,solver=None, options={'reltol':1e-9})



for i in range(len(xdata)-128):
    if r[i]<r[i+1] and 0.010<r[i]<0.011:
        onsetx.append(i/128)
        onsety.append(r[i])

    if r[i]>r[i+1] and 0.010<r[i]<0.011:
        offsetx.append(i/128)
        offsety.append(r[i])

length = len(onsetx)


onsetx,onsety = remove_close_values(onsetx,onsety, 0.7)
offsetx,offsety = remove_close_values(offsetx,offsety, 0.7)


plt.subplot(221)
plt.title("Skin Conductance")
plt.xlabel('Time (s)')
plt.ylabel('μS')
plt.plot(xdata,ydata,label = "Skin Conductance")
plt.legend()

for i in range(len(onsetx)):
    while i < len(offsetx):
        diff = abs(onsetx[i]-offsetx[i])
        if diff<1:
            onsetx.remove(onsetx[i])
            offsetx.remove(offsetx[i])
            onsety.remove(onsety[i])
            offsety.remove(offsety[i])
        i +=1
  





plt.subplot(222)
plt.title("Phasic Component")
plt.xlabel('Time (s)')
plt.ylabel('μS')
plt.plot(xdata, r,label = "phasic component")
plt.legend()

plt.scatter(onsetx, onsety)
plt.scatter(offsetx,offsety)



plt.subplot(223)
plt.title("Tonic Component")
plt.xlabel('Time (s)')
plt.ylabel('μS')
plt.plot(xdata,t,label = "tonic component")


print("coefficients of tonic spline: ",l,"offset and slope of the linear drift term",d,"model residuals",e)
print("onset and offset is: ",onsetx,onsety)
plt.plot(xdata,e,label = "model residuals")
plt.legend()

plt.tight_layout(pad = 2.0)
print("the length is ",len(ydata))
plt.show()


""" Arguments:
       y: observed EDA signal (we recommend normalizing it: y = zscore(y))
       delta: sampling interval (in seconds) of y
       tau0: slow time constant of the Bateman function
       tau1: fast time constant of the Bateman function
       delta_knot: time between knots of the tonic spline function
       alpha: penalization for the sparse SMNA driver
       gamma: penalization for the tonic spline coefficients
       solver: sparse QP solver to be used, see cvxopt.solvers.qp
       options: solver options, see:
                http://cvxopt.org/userguide/coneprog.html#algorithm-parameters

    Returns (see paper for details):
       r: phasic component
       p: sparse SMNA driver of phasic component
       t: tonic component
       l: coefficients of tonic spline
       d: offset and slope of the linear drift term
       e: model residuals
       obj: value of objective function being minimized (eq 15 of paper)
    """