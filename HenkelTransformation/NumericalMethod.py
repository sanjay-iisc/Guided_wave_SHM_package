import numpy as np
from numpy import pi
from scipy.special import jv
import scipy.special as scipybessel
def f1(x):
    #1/(1+x**2)
    return jv(0.5,x)*(x**(-0.5))#*(x/(x**2 + 1))


def trapezoidal(x0,xn,n):
    # calculating step size
    h = (xn - x0) / n
    # Finding sum 
    integration = f1(x0) + f1(xn)
    for i in range(1,n):
        k = x0 + i*h
        integration = integration + 2 * f1(k)
    integration = integration * h/2
    return  integration
def compositesTrapezoidal(x0,xn,n, N=40):
    xi=x0
    xf=xn
    temp=0
    for i in np.arange(1,N,1) :
        xf= i*(xn/N)
        temp=temp+trapezoidal(xi,xf,n)
        xi=xf
    return temp



def gauss_quadrature_integration(n):
    s = 0
    xgauleg, wgauleg = np.polynomial.laguerre.laggauss(n)
    for i in range(1,n,1):
       s = s+ xgauleg[i]*xgauleg[i]*wgauleg[i]
    return s

def pythonBasedtrap(x0,xn,n):
    # calculating step size
    h = (xn - x0) / n
    t = np.arange(x0,xn+h,h)
    return np.trapz(f1(t),x=t, dx=h)

if __name__=='__main__':
    n =1000
    xi=0.01
    xf=700
    # print('inter={}'.format(trapezoidal(xi,xf,N)))
    # print('inter={}'.format(pythonBasedtrap(xi,xf,N)))
    print('inter={}'.format(compositesTrapezoidal(xi,xf,n,N=100)))