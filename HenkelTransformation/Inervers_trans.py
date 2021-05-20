import numpy as np                                                        # Numpy
from powerbox import dft                                                  # powerbox for DFTs (v0.6.0+)
from hankel import HankelTransform                            # Hankel
from scipy.interpolate import InterpolatedUnivariateSpline as spline      # Splines
import matplotlib.pyplot as plt      
from scipy.special  import jv
from scipy import integrate
from scipy.integrate import simps
import hankel
from scipy.interpolate import interp1d

f = lambda x: jv(1,x*5)/x
k = np.logspace(-1,1,100)  
ht = HankelTransform(nu=1,N=1024*20,h=1e-6) 

hhat = ht.transform(f,k,ret_err=False)  
# dta=hankel.get_h(
#     f, nu=0,
#     K= np.array([1, 100]),
#     cls=HankelTransform
# )
# print(dta)
plt.figure()
plt.plot(k,hhat)
plt.show()



