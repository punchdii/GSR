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




data = pd.read_csv("accy50hz.csv")

ydata = data['Shimmer_3B52_Accel_LN_Y_CAL']
for i in range(len(ydata)):
    ydata[i] = ydata[i]-2

ydata  = ydata[98:140]
ydata = median_filter(ydata,1)
xdata = np.linspace(0,1.04,52)

plt.title("Accelaration")
plt.xlabel('Time (s)')
plt.ylabel('y-accelaration')
#
#plt.ylim()
plt.figure(1,dpi=120)
plt.plot(xdata,ydata,label = "Experiment")



#define Functions
def func(x,a,b,c):
    return a*x**2+b*x+c
constants = curve_fit(func,xdata,ydata)
a_fit= constants[0][0]
b_fit= constants[0][1]
c_fit= constants[0][2]



fit = []
for i in xdata:
    fit.append(func(i,a_fit,b_fit,c_fit))



a = a_fit
b = b_fit
c = c_fit

def f(x):
    return a*x**2+b*x+c

x= []
y = []
for i in xdata:
    x.append(i)
    y.append(f(i))
plt.plot(x,y,label = "acc app")



vx = []
vy = []
for t in xdata:
    integ = quad(f,0,t)
    vx.append(t)
    vy.append(integ[0])
  
result = quad(f,0,0.44)
print("the integration for accelaration which should be the velocity is",result[0])

#approcimate velocity

def cubicf(x,a,b,c,d):
    return a*x*x*x+b*x**2+c*x+d

print(len(vx),len(vy))
constants = curve_fit(cubicf,vx,vy)
a_fit= constants[0][0]
b_fit= constants[0][1]
c_fit= constants[0][2]
d_fit= constants[0][3]




cubicx =[]
cubicy = []
for i in xdata:
    cubicx.append(i)
    cubicy.append(cubicf(i,a_fit,b_fit,c_fit,d_fit))


a = a_fit
b = b_fit
c = c_fit
d = d_fit
def cubicfapp(x):
    return a*x*x*x+b*x**2+c*x+d

dx = []
dy = []
for t in xdata:
    result = quad(cubicfapp,0,t)
    dx.append(t)
    dy.append(result[0])

final_result = str(result[0]*100)+"cm"
print("the integration for velocity which should be the displacement is",final_result)

plt.plot(dx,dy,label = "displacement")
plt.plot(cubicx,cubicy,label = "velocity_app")
plt.plot(vx,vy,label = "velocity")
plt.legend()
plt.show()

#perr = np.sqrt(np.diag(pcov))

#exampledata = np.array(rawdata[1:],dtype = np.float)
