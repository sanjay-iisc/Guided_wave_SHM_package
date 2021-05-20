from scipy.special import yv,jv
from scipy.integrate import simps
from mpmath import fp as mpm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from hankel import HankelTransform, SymmetricFourierTransform
import hankel

def sinc(x):
    y=np.sin(x*5) / x
    interp1d(x,y,fill_value=(0, 0) ,bounds_error=False)
    return 

def hankel_transform_of_sinc(v,gamma):
    ht = np.zeros_like(v)
    ht[v < gamma] = (v[v < gamma] ** p * np.cos(p * np.pi / 2)
                     / (2 * np.pi * gamma * np.sqrt(gamma ** 2 - v[v < gamma] ** 2)
                        * (gamma + np.sqrt(gamma ** 2 - v[v < gamma] ** 2)) ** p))
    ht[v >= gamma] = (np.sin(p * np.arcsin(gamma / v[v >= gamma]))
                      / (2 * np.pi * gamma * np.sqrt(v[v >= gamma] ** 2 - gamma ** 2)))
    return ht


ht = HankelTransform(nu=1,N=1024*3,h=1e-5)
k = np.logspace(-1,1,1000)
f_new = ht.transform(sinc, k, False, inverse=True)
plt.figure()
plt.plot(k,f_new)
plt.show()