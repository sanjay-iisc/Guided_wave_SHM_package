from scipy.special import yv,jv
from scipy.integrate import simps
from mpmath import fp as mpm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from hankel import HankelTransform, SymmetricFourierTransform
import hankel

def sinc(x):
    
    return np.where( x < 20 , np.sin(x*5) / x,0)

x=np.linspace(0,100,1000)
p=1
def hankel_transform_of_sinc(v,gamma):
    ht = np.zeros_like(v)
    ht[v < gamma] = (v[v < gamma] ** p * np.cos(p * np.pi / 2)
                     / (2 * np.pi * gamma * np.sqrt(gamma ** 2 - v[v < gamma] ** 2)
                        * (gamma + np.sqrt(gamma ** 2 - v[v < gamma] ** 2)) ** p))
    ht[v >= gamma] = (np.sin(p * np.arcsin(gamma / v[v >= gamma]))
                      / (2 * np.pi * gamma * np.sqrt(v[v >= gamma] ** 2 - gamma ** 2)))
    return ht


ht = HankelTransform(nu=p,N=1024*5,h=1e-4)
k = np.logspace(-1,2,100)
f_new = ht.transform(sinc, k, False, inverse=True)
plt.figure()
plt.plot(x,sinc(x))
# plt.plot(k,hankel_transform_of_sinc(k,5))

plt.figure()
plt.plot(k,f_new)
plt.show()