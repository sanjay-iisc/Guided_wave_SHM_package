import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import Hybridmodel as HM
import Figure_plot as graph
import matplotlib.pyplot as plt
import scipy.special 
from scipy.special  import jv
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import count
f=700e3
# Importing the wave stress
Freq = np.arange(5, 1000, 5)*1e3 # Hz
p= np.argmin(abs(Freq-f))
Rz_waveNumber=pd.read_csv("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\data_stress_RZ_waveNumber.csv")
stress_KRz=Rz_waveNumber['sigma_RZ[N/mm^2] '+'F='+str(int(Freq[p-1]*1e-3))+' [KHz]']
K_rz=Rz_waveNumber['K[rad/mm]']*1e3
K=np.linspace(10,3000,100)
stress_KRz=interp1d(K_rz, stress_KRz)(K)

#############-----
Model=HM.t_w()
Ks=Model._tipDisp._equations.K[:,2]
Ka=Model._tipDisp._equations.K[:,1]
x=Model._tipDisp._equations.Freq
a=Model._tipDisp._equations.a
# Admittance term
Aw=Model.constan_term(isPlotting=False)
fAw=interp1d(x,Aw)
# t_r=x1*a*jv(1,K*a*x3)+x2*fAw(f)*a*jv(2,K*a*x3)/K
def demo_func(p):
    x1,x2,x3,x4=p
    t_r=x1*fAw(f)*jv(x2,K*a*x3)*(K*a)**x4#x1*a*jv(1,K*a*x2)+x3*fAw(f)*a*jv(2,K*a*x2)/K
    return np.square((stress_KRz-t_r)).sum()#/np.square((stress_KRz)).sum()

