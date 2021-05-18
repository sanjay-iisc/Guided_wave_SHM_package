
import sys
sys.path.append("./")
import GuidedWaveModelling.Hybridmodel as HM
import GuidedWaveModelling.PinForce as PF
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
Model=HM.t_w()
Ks=Model._tipDisp._equations.K[:,2]
Ka=Model._tipDisp._equations.K[:,1]
x=Model._tipDisp._equations.Freq
a=Model._tipDisp._equations.a
# Admittance term
Aw=Model.constan_term(isPlotting=False)
fAw=interp1d(x,Aw)

for nFreq in np.arange(125, 200, 10)*1e3:
    Rz_waveNumber=pd.read_csv("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\data_stress_RZ_waveNumber.csv")
    stress_KRz=Rz_waveNumber['sigma_RZ[N/mm^2] '+'F='+str(int(nFreq*1e-3))+' [KHz]']
    K_rz=Rz_waveNumber['K[rad/mm]']*1e3
    K=np.linspace(10,3000,100)
    stress_KRz=interp1d(K_rz, stress_KRz)(K)
    
    def demo_func(p):
        x1=p[0]#,x2,x3,x4=p
        # print(p)
        x2=p[1]#1.17
        x3=p[2]
        x4=p[3]
        t_r=x1*fAw(nFreq)*jv(x2,K*a*x3)*(K*a)**x4#x1*fAw(nFreq)*0.13*jv(x2,K*a*x3)*(K*a)**x4#x1*a*jv(1,K*a*x2)+x3*fAw(f)*a*jv(2,K*a*x2)/K
        return  np.square((stress_KRz-t_r)).sum()/np.square((stress_KRz)).sum()


    path="E:\Work\Work\\Nicolas_opti_results_2\RR\\"+"F_"+str(int(nFreq*1e-3))+'_KHz'
    GA.GeneticAlgorithm_Base._get_userInputs(demo_func,dim=4,max_intr=500,population_size=1000,dir_name_save=path)
    # # print('sa'
    GA1=GA.GA_strat(ifsaveReport=True)
    GA1.RUN()



# P=[0.06147741967155907, 1.2053076692798892, 0.9619713094471052, -0.21348464355949529] #[-0.002297306286524377, 0.10797527486310032, 1, -0.3399429031744833] #[-0.061769510606423716, 0.4487918853084113, 0.7094666353599042, -0.5102433232527348]#[-0.03528800612320042, -0.3360330302220776, 0.5849591589741946, 0.23164138981744262]#[-0.017615108654933626, 1.021239576177761]#[-0.026087045026607403, 1.7974349624301966]
# plt.figure()
# plt.plot(K,stress_KRz, label ='FEM')
# plt.plot(K,demo_func(P), label ='Optimized')
# plt.legend()
# plt.show()
