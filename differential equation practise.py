# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:10:16 2020

@author: edwar
"""


import numpy as np
integer=float(input('input a positive integer'))
if integer>0:
    a=np.array([])
    while a.size<integer:
        b=float(input('input another integer'))
        a=np.append(a,b)


    abar=np.mean(a)
    amed=np.median(a)
    asum=np.sum(a)
    print('your integers have a mean of ',abar,', a median of ',amed,' and the sum equals ',asum)
else:
    print('not a positive integer')
#%%
import numpy as np
import matplotlib.pyplot as plt
r=np.random.randint(1,100,1200)
b = np.arange(1, 100, 1) 
plt.hist(r,bins=b)
a=np.histogram(r,bins=b)
#%%
#calculating value for pi
import matplotlib.pyplot as plt
import numpy as np
a=np.array([])
n=0
piaprox=0
nmax=300
for n in range(0,nmax+1):
    term= ((-1)**n)/(2 *n + 1)
    piaprox=term+piaprox
    a=np.append(a,piaprox)
    
piaprox=4*piaprox    
print('pi=',piaprox)

narray=np.arange(0,301)
plt.plot(narray,4*a)
#%%
#squaring inputs
import numpy as np
ksqrvector=np.array([])
n=int(input('input a positive integer'))
nvector=np.arange(0,n+1)
ksqr=(nvector*(nvector+1)*(2*nvector+1))/6
a=sum(ksqr)
print('value of the sum of the integers squared up to n=',n,' is ',a)
#%%
#multiple plots
import numpy as np
import matplotlib.pyplot as plt
E=np.linspace(0,1.5e-15,100)
k=1.38e-23
for T in [1e-7,2e-7,3e-7]:
    MBdist=(2/np.sqrt(np.pi))*((2*k*T)**-(3/2))*(E**(1/2))*np.exp(-E/k*T)
    plt.plot(E,MBdist)
plt.xlabel('$E (J)$')
plt.ylabel('$F(E)$')
plt.legend(('$T1$', '$T2$','$T3$'), fontsize='14')
#%%
import time
import numpy as np
import matplotlib.pyplot as plt
F=np.array([])
X=np.array([])
r=3.599459999999999999
x=0.78
n=0
N=np.array([])
for n in range(0,100):
    n=n+1
    x=r*x*(1-x)
    f=r*x*(1-x)
    F=np.append(F,f)
    X=np.append(X,x)
    N=np.append(N,n)
plt.figure(1)
plt.plot(N,X,'g*-')
plt.figure(2)
plt.plot(F,X,'g*-')

#%%
#Solving 1st order ODE
import numpy as np
import matplotlib.pyplot as plt

k=11
m=0.1
x=0.05
t=0
dt=0.01
v=0
V=np.array([])
X=np.array([])
T=np.array([])

while t<2 : 
    a=-k*x/m
    v=v+a*dt
    x=x+v*dt
    X=np.append(X,x)
    V=np.append(V,v)
    t=t+dt
    T=np.append(T,t)
plt.plot(T,X)
plt.ylabel('$x (t)$')
plt.xlabel('$t$')
#%%
#solving simple 2nd order ODE
import numpy as np
import matplotlib.pyplot as plt

w=0.5
y=0.5
x=0
dx=0.001

W=np.array([])
X=np.array([])
Y=np.array([])

while x<20:
    d2y=np.sin(x)-y
    w=d2y*dx+w
    y=w*dx+y
    Y=np.append(Y,y)
    X=np.append(X,x)
    x=x+dx
plt.plot(X,Y)
plt.ylabel('$y (x)$')
plt.xlabel('$x$')   








