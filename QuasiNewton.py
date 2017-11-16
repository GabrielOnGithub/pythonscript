# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 02:13:15 2017

@author: Gabriel
"""

#%% https://nlperic.github.io/line-search/

import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# bisection search of wolfe condition
def func(x):
    return 100*np.square(np.square(x[0])-x[1]) + np.square(x[0]-1)

def dfunc(x):
    df1 = 400*x[0]*(np.square(x[0])-x[1])+2*(x[0]-1)
    df2 = -200*(np.square(x[0])-x[1])
    return np.array([df1, df2])

def wolfe(valf, direction, max_iter):
    (alpha, beta, step, c1, c2) = (0, 1000, 5.0, 0.15, 0.3)
    i = 0
    stop_iter = 0
    stop_val = valf
    minima = 0
    val = []
    objectf = []
    val.append(valf)
    objectf.append(func(valf))
    while i <= max_iter:
        # first confition
        leftf = func(valf + step*direction)
        rightf = func(valf) + c1*dfunc(valf).dot(direction)
        if leftf > rightf:
            beta = step
            step = .5*(alpha + beta)
            val.append(valf+step*direction)
            objectf.append(leftf)
        elif dfunc(valf + step*direction).dot(direction) < c2*dfunc(valf).dot(direction):
            alpha = step
            if beta > 100:
                step = 2*alpha
            else:
                step = .5*(alpha + beta)
            val.append(valf+step*direction)
            objectf.append(leftf)
        else:
            val.append(valf+step*direction)
            objectf.append(leftf)
            break
        i += 1
        stop_val = valf + step*direction
        stop_iter = i
        minima = func(stop_val)
    print(val, objectf)
    return stop_val, minima, stop_iter, step, val, objectf

start = np.array([.6, .5])
dirn = np.array([-.3, -.4])
converge_value, minimal, no_iter, size, val, objectf = wolfe(start, dirn, 30)
print("The value, minimal and iterations needed are " + str(converge_value) + ", " \
+ str(minimal) + ", " + str(no_iter) + ', ' + str(size))
x = np.array([i[0] for i in val])
y = np.array([i[1] for i in val])
z = np.array(objectf)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, label='Wolfe Rule')
ax.legend()
plt.savefig('./res/wolfe.jpg')

#%% Armijo

import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# the objective function
def func(x):
    return 100*np.square(np.square(x[0])-x[1])+np.square(x[0]-1)

# first order derivatives of the function
def dfunc(x):
    df1 = 400*x[0]*(np.square(x[0])-x[1])+2*(x[0]-1)
    df2 = -200*(np.square(x[0])-x[1])
    return np.array([df1, df2])

# the armijo algorithm
def armijo(valf, grad, niters):
    #beta = random.random()
    #sigma = random.uniform(0, .5)
    beta = 0.25
    sigma = 0.25
    (miter, iter_conv) = (0, 0)
    conval = [0,0]
    val = []
    objectf = []
    val.append(valf)
    objectf.append(func(valf))
    while miter <= niters:
        leftf = func(valf+np.power(beta, miter)*grad)
        rightf = func(valf) + sigma*np.power(beta, miter)*dfunc(valf).dot(grad)
        if leftf-rightf <= 0:
            iter_conv = miter
            conval = valf+np.power(beta, iter_conv)*grad
            break
        miter += 1
        val.append(conval)
        objectf.append(func(conval))
    return conval, func(conval), iter_conv, val, objectf

# initialization
start = np.array([-.3, .1])
direction = np.array([1, -2])
maximum_iterations = 30

converge_value, minimal, no_iter, val, objf = armijo(start, direction, maximum_iterations)
print("The value, minimal and number of iterations are " + str(converge_value) + \
    ", " + str(minimal) + ", " + str(no_iter))
x = np.array([i[0] for i in val])
y = np.array([i[1] for i in val])
z = np.array(objf)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, label='Armijo Rule')
ax.legend()
plt.savefig('./res/armijo.jpg')

#%%Newton 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv

def func(x):
    return 100*np.square(np.square(x[0])-x[1])+np.square(x[0]-1)

# first order derivatives of the function
def dfunc(x):
    df1 = 400*x[0]*(np.square(x[0])-x[1])+2*(x[0]-1)
    df2 = -200*(np.square(x[0])-x[1])
    return np.array([df1, df2])

def invhess(x):
    df11 = 1200*np.square(x[0])-400*x[1]+2
    df12 = -400*x[0]
    df21 = -400*x[0]
    df22 = 200
    hess = np.array([[df11, df12], [df21, df22]])
    return inv(hess)

def newton(x, max_int):
    miter = 1
    step = .5
    vals = []
    objectfs = []
    # you can customize your own condition of convergence, here we limit the number of iterations
    while miter <= max_int:
        vals.append(x)
        objectfs.append(func(x))
        temp = x-step*(invhess(x).dot(dfunc(x)))
        if np.abs(func(temp)-func(x))>0.01:
            x = temp
        else:
            break
        print(x, func(x), miter)
        miter += 1
    return vals, objectfs, miter

start = [5, 5]
val, objectf, iters = newton(start, 50)

x = np.array([i[0] for i in val])
y = np.array([i[1] for i in val])
z = np.array(objectf)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, label='newton method')
plt.savefig('./res/newton.jpg')

#%% Quasi Newton
import matplotlib.pyplot as plt
import numpy as np
import random
import math

delta = 0.1
minXY=-5.0
maxXY=5.0
nContour=50
alpha=0.001

def Jacob(state):
    u"""
    jacobi matrix of Himmelblau's function
    """
    x=state[0,0]
    y=state[0,1]
    dx=4*x**3+4*x*y-44*x+2*x+2*y**2-14
    dy=2*x**2+4*x*y+4*y**3-26*y-22
    J=np.matrix([dx,dy]).T
    return J

def HimmelblauFunction(x,y):
    u"""
    Himmelblau's function
    see Himmelblau's function - Wikipedia, the free encyclopedia 
    http://en.wikipedia.org/wiki/Himmelblau%27s_function
    """
    return (x**2+y-11)**2+(x+y**2-7)**2

def CreateMeshData():
    x = np.arange(minXY, maxXY, delta)
    y = np.arange(minXY, maxXY, delta)
    X, Y = np.meshgrid(x, y)
    Z=[HimmelblauFunction(x,y) for (x,y) in zip(X,Y)]
    return(X,Y,Z)

def QuasiNewtonMethod(start,Jacob):
    u"""
    Quasi Newton Method Optimization
    """

    result=start
    x=start

    H= np.identity(2)
    preJ=None
    preG=None

    while 1:
        J=Jacob(x)

        sumJ=abs(np.sum(J))
        if sumJ<=0.01:
            print("OK")
            break

        grad=-np.linalg.inv(H)*J
        x+=alpha*grad.T
        
        result=np.vstack((result,np.array(x)))

        if preJ is not None:
            y=J-preJ
            H=H+(y*y.T)/(y.T*preG)-(H*preG*preG.T*H)/(preG.T*H*preG)

        preJ=J
        preG=(alpha*grad.T).T

    return result

# Main
start=np.matrix([random.uniform(minXY,maxXY),random.uniform(minXY,maxXY)])

result=QuasiNewtonMethod(start,Jacob)
(X,Y,Z)=CreateMeshData()
CS = plt.contour(X, Y, Z,nContour)

plt.plot(start[0,0],start[0,1],"xr");

optX=result[:,0]
optY=result[:,1]
plt.plot(optX,optY,"-r");

plt.show()

#%%
#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random

delta = 0.1
minXY=-5.0
maxXY=5.0
nContour=50
alpha=0.01

def Hessian(state):
    u"""
    Hessian matrix of Himmelblau's function
    """
    x=state[0]
    y=state[1]
    dxx=12*x**2+4*y-42;
    dxy=4*x+4*y
    dyy=4*x+12*y**2-26
    H=np.array([[dxx,dxy],[dxy,dyy]])
    return H
    

def Jacob(state):
    u"""
    jacobi matrix of Himmelblau's function
    """
    x=state[0]
    y=state[1]
    dx=4*x**3+4*x*y-44*x+2*x+2*y**2-14
    dy=2*x**2+4*x*y+4*y**3-26*y-22
    J=[dx,dy]
    return J

def HimmelblauFunction(x,y):
    u"""
    Himmelblau's function
    see Himmelblau's function - Wikipedia, the free encyclopedia 
    http://en.wikipedia.org/wiki/Himmelblau%27s_function
    """
    return (x**2+y-11)**2+(x+y**2-7)**2

def CreateMeshData():
    x = np.arange(minXY, maxXY, delta)
    y = np.arange(minXY, maxXY, delta)
    X, Y = np.meshgrid(x, y)
    Z=[HimmelblauFunction(x,y) for (x,y) in zip(X,Y)]
    return(X,Y,Z)

def NewtonMethod(start,Jacob):
    u"""
    Newton Method Optimization
    """

    result=start
    x=start

    while 1:
        J=Jacob(x)
        H=Hessian(x)
        sumJ=sum([abs(alpha*j) for j in J])
        if sumJ<=0.01:
            print("OK")
            break

        grad=-np.linalg.inv(H).dot(J) 
        print(grad)

        x=x+[alpha*j for j in grad]
        
        result=np.vstack((result,x))

    return result

# Main
start=np.array([random.uniform(minXY,maxXY),random.uniform(minXY,maxXY)])

result=NewtonMethod(start,Jacob)
(X,Y,Z)=CreateMeshData()
CS = plt.contour(X, Y, Z,nContour)
#  plt.clabel(CS, inline=1, fontsize=10)
#  plt.title('Simplest default with labels')

plt.plot(start[0],start[1],"xr");

optX=[x[0] for x in result]
optY=[x[1] for x in result]
plt.plot(optX,optY,"-r");

plt.show()
#%%Steepest descent

import matplotlib.pyplot as plt
import numpy as np
import random

delta = 0.1
minXY = -5.0
maxXY = 5.0
nContour = 50
alpha = 0.01


def Jacob(state):
    u"""
    jacobi matrix of Himmelblau's function
    """
    x = state[0, 0]
    y = state[0, 1]
    dx = 4 * x ** 3 + 4 * x * y - 44 * x + 2 * x + 2 * y ** 2 - 14
    dy = 2 * x ** 2 + 4 * x * y + 4 * y ** 3 - 26 * y - 22
    J = np.matrix([dx, dy])
    return J


def HimmelblauFunction(x, y):
    u"""
    Himmelblau's function
    see Himmelblau's function - Wikipedia, the free encyclopedia
    http://en.wikipedia.org/wiki/Himmelblau%27s_function
    """
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def ConstrainFunction(x):
    return (2.0 * x + 1.0)


def CreateMeshData():
    x = np.arange(minXY, maxXY, delta)
    y = np.arange(minXY, maxXY, delta)
    X, Y = np.meshgrid(x, y)
    Z = [HimmelblauFunction(ix, iy) for (ix, iy) in zip(X, Y)]
    return(X, Y, Z)


def SteepestDescentMethod(start, Jacob):
    u"""
    Steepest Descent Method Optimization
    """

    result = start
    x = start

    while 1:
        J = Jacob(x)
        sumJ = np.sum(abs(alpha * J))
        if sumJ <= 0.01:
            print("OK")
            break

        x = x - alpha * J
        result = np.vstack((result, x))

    return result


# Main
start = np.matrix([random.uniform(minXY, maxXY), random.uniform(minXY, maxXY)])

result = SteepestDescentMethod(start, Jacob)
(X, Y, Z) = CreateMeshData()
CS = plt.contour(X, Y, Z, nContour)

Xc = np.arange(minXY, maxXY, delta)
Yc = [ConstrainFunction(x) for x in Xc]

plt.plot(start[0, 0], start[0, 1], "xr")

plt.plot(result[:, 0], result[:, 1], "-r")

plt.axis([minXY, maxXY, minXY, maxXY])
plt.show()

#%%