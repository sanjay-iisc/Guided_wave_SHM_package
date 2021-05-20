import sys
sys.path.append("./")
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import os 
import pandas as pd
from scipy.special  import jv
import GuidedWaveModelling.Figure_plot as graph
from scipy import integrate
from pyhank import HankelTransform

def stressRZ(p):
    path='K:\\LMC\\Sanjay\\Comsolresults\\NicolasResults\\stressnew2' 
    fileName= 'Stress_RZ_'+str(p)+'.csv' # the file is starting from 1
    df = pd.read_csv(os.path.join(path , fileName), skiprows=8)
    ##
    columns =df.columns
    df.columns=['Radius (mm)', 'thickness (mm)', 'stress (N/m^2)', 'f0']
    SigmaRz= df['stress (N/m^2)'].str.replace('i','j').apply(lambda x: np.complex128(x))*1e-6# N/mm2
    R=df['Radius (mm)']

    dK = 2*np.pi/(R[1]-R[0])
    K= np.arange(10,1e4)


    return R, np.real(SigmaRz)




if __name__=='__main__':
    # r = np.linspace(0, 100, 1024)
    # f = np.zeros_like(r)
    # f[1:] = jv(1, r[1:]) 
    # f[r == 0] = 0.5

    transformer = HankelTransform(order=0, max_radius=100, n_points=1024)
    f = jv(1.5, transformer.r*5) #/ transformer.r
    ht = transformer.qdht(f)
    plt.figure()
    plt.plot(transformer.kr, ht)
    # plt.xlim([0, 5])
    plt.xlabel('Radial wavevector /m$^{-1}$')
    plt.show()
    
