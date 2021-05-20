import sys
sys.path.append("./")
import GuidedWaveModelling.Hybridmodel as HM
import GuidedWaveModelling.Figure_plot as graph
import matplotlib.pyplot as plt
import scipy.special 
from scipy.special  import jv
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import numpy as np
import main as GA
#----------------------------------
plt.close('All')
Model=HM.t_w()
Ks=Model._tipDisp._equations.K[:,2]
Ka=Model._tipDisp._equations.K[:,1]
x=Model._tipDisp._equations.Freq
a=Model._tipDisp._equations.a
# Admittance term
Aw=Model.constan_term(isPlotting=False)
fAw=interp1d(x,Aw)

def stress_fem(nFreq,axes):
    ZZ_waveNumber=pd.read_csv("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\data_stress_ZZ_waveNumber.csv")
    stress_KZZ=ZZ_waveNumber['sigma_ZZ[N/mm^2] '+'F='+str(int(nFreq*1e-3))+' [KHz]']
    K_ZZ=ZZ_waveNumber['K[rad/mm]']*1e3
    K=np.linspace(10,3000,100)
    stress_KZZ=Spline(K_ZZ, stress_KZZ)(K)

    x1=0.0695
        # print(p)
    x2=1.17
    x3=0.92
    x4=0.45
    aproximated=0.13*fAw(nFreq)*x1*jv(x2,K*a*x3)*(K*a)**x4
    graph.figureplot(K,stress_KZZ,ax=axes, label ='FEM')
    graph.figureplot(K,aproximated,ax=axes, label='Appro')

if __name__=='__main__':
    fig,axes=plt.subplots(1,1)
    for f in np.arange(750,800,1000)*1e3:
        stress_fem(f,axes)

    plt.show()

    