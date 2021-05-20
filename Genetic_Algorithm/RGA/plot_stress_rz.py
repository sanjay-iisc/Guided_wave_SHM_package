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
Model._tipDisp._equations.plottingWaveNumber()
# Admittance term
Aw=Model.constan_term(isPlotting=False)
fAw=interp1d(x,Aw)
KA0=interp1d(x,Ka)
# print(x)
def stress_fem(nFreq):
    RZ_waveNumber=pd.read_csv("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\data_stress_RZ_waveNumber.csv")
    stress_KRZ_1=RZ_waveNumber['sigma_RZ[N/mm^2] '+'F='+str(int(nFreq*1e-3))+' [KHz]']
    K_RZ=RZ_waveNumber['K[rad/mm]']*1e3
    print(min(K_RZ))
    K=np.linspace(10,3000,1000)
    stress_KRZ=interp1d(K_RZ, stress_KRZ_1)(K)
    
    trrA=interp1d(K_RZ, stress_KRZ_1)(abs(KA0(nFreq)))
    print(trrA)
    print(fAw(nFreq))
    x1=0.0494014621871498
        # print(p)
    x2=1.08739011205002
    x3=0.941755350437559
    x4=-0.164273272372193
    aproximated=x1*fAw(nFreq)*jv(x2,K*a*x3)*(K*a)**x4
    fig,axes=plt.subplots(1,1)
    graph.figureplot(K,stress_KRZ,ax=axes, label ='FEM')
    graph.figureplot(K,aproximated,ax=axes, label='Appro')
    plt.scatter(abs(KA0(nFreq)),trrA)





if __name__=='__main__':
    stress_fem(15e3)
    plt.show()

    