import sys
sys.path.append("./")
import numpy as np
import pandas as pd
import GuidedWaveModelling.Figure_plot as graph
import GuidedWaveModelling.PinForce as PF
import matplotlib.pyplot as plt
import os
import GuidedWaveModelling.Hybridmodel as HM
from scipy.interpolate import interp1d
from scipy.special  import jv
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
Model=HM.t_w()
Ks=Model._tipDisp._equations.K[:,2]
Ka=Model._tipDisp._equations.K[:,1]
freqModel=Model._tipDisp._equations.Freq
a=Model._tipDisp._equations.a
# Admittance term
Aw=Model.constan_term(isPlotting=False)
fAw=interp1d(freqModel,Aw)

DataRR= pd.read_csv("K:\LMC\Sanjay\Code\Optimization\optimization_stress\Optimized_stress_const_RR\\stress_hyperperameter_RR.csv")
Data= pd.read_csv("K:\LMC\Sanjay\Code\Optimization\optimization_stress\Optimized_stress_const_RR\\stress_hyperperameter_ZZ.csv")
Data_mean=Data.mean(axis=0)
print(Data_mean)
print(Data.var(axis=0))
KS0=interp1d(freqModel,Ks)(Data['Freq[Hz]'])
KA0=interp1d(freqModel,Ka)(Data['Freq[Hz]'])
tw=interp1d(DataRR['Freq[Hz]'],DataRR['tr'])(Data['Freq[Hz]'])
t_z_s0=tw*Data['zeta']*Data['A(w)']*jv(Data['alpha'],KS0*Data['gamma']*a)*(KS0*a)**Data['beta']
#
# t_r_s0=Data['tr']*Data['A(w)']*jv(1,KS0*Data['gamma']*a)
t_z_s0_mean=tw*Data_mean['zeta']*Data['A(w)']*jv(Data_mean['alpha'],KS0*Data_mean['gamma']*a)*(KS0*a)**Data_mean['beta']

t_z_a0=tw*Data['zeta']*Data['A(w)']*jv(Data['alpha'],KA0*Data['gamma']*a)*(KA0*a)**Data['beta']

t_z_a0_mean=tw*Data_mean['zeta']*Data['A(w)']*jv(Data_mean['alpha'],KA0*Data_mean['gamma']*a)*(KA0*a)**Data_mean['beta']
# t_r_a0_mean=Data['tr']*Data['A(w)']*jv(1.11,KA0*0.92*a)*(KA0*a)**-0.22
#-----PF Model
PF_model=PF.Displacement_Field_PF()  
UPF,pf_tS,pf_tA=PF_model.PF_Displacement(isPlotting=False)

#--------------FEM -Stress
FemFreq = np.arange(5, 1000, 5)*1e3
t11_S0=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\\calcul_modal\\NicolasPlate\stressSZ.npy"))(Data['Freq[Hz]'])
t11_A0=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\\calcul_modal\\NicolasPlate\stressAZ.npy"))(Data['Freq[Hz]'])


pathsave ="E:\PPT\Presentation\\_01062021_ppt\Figure\\"
fig,axes=plt.subplots(4,1, sharex=True)
graph.figureplot(Data['Freq[Hz]'],Data['alpha'],ax=axes[0], linestyle='--',marker='o',label=r'$\alpha_z$',c='k', title='Hyper-Parameters-'+r'$\tau_{zz}$', markersize=2)


# fig,axes=plt.subplots(1,1)
graph.figureplot(Data['Freq[Hz]'],Data['gamma'],ax=axes[1], linestyle='--',marker='o',label=r'$\gamma_z$',c='r', markersize=2)

# fig,axes=plt.subplots(1,1)
graph.figureplot(Data['Freq[Hz]'],Data['beta'],ax=axes[2], linestyle='--',marker='o',label=r'$\beta_z$',c='b', markersize=2)
graph.figureplot(Data['Freq[Hz]'],Data['zeta'],ax=axes[3], linestyle='--',marker='o',label=r'$\zeta_z$',c='g', markersize=2,path=pathsave,filename='hyper_tzz')

# fig,axes=plt.subplots(1,1)
# graph.figureplot(Freq*1e3,(X0),ax=axes, linestyle='--',marker='*',title=r'$\tau_r$')
# graph.figureplot(Data['Freq[Hz]'],(X0),ax=axes, linestyle='--',marker='*',title=r'$\tau_r$')
# graph.figureplot(np.arange(5, 1000, 20)*1e3,(tr),ax=axes, linestyle='--',marker='*',title=r'$\tau_r$', label='Era')

fig,axes=plt.subplots(1,2, sharey=True)
#-------------S0------------------------

graph.figureplot(Data['Freq[Hz]'],abs(t_z_s0),ax=axes[0], linestyle='-',c='b',ylabel=r'$\tau_z[N]$',title='S0', label='HM-Optimized')
graph.figureplot(Data['Freq[Hz]'],abs(t_z_s0_mean),ax=axes[0], linestyle='--',marker='*',c='r',ylabel=r'$\tau_z[N]$', title='S0',label='HM-Optimized-mean', alpha=1,markersize=2)
# graph.figureplot(PF_model._equations.Freq, abs(pf_tS[:,0]),ax=axes[0], linestyle='-',c='k',ylabel=r'$\tau_r[N]$', title='S0',label='PF')
graph.figureplot(Data['Freq[Hz]'], abs(t11_S0),ax=axes[0], linestyle='None',marker='o',c='k',ylabel=r'$\tau_z[N]$', title='S0',label='FEM', alpha=1,markersize=2)
#-----A0------------------------------------------
graph.figureplot(Data['Freq[Hz]'],abs(t_z_a0),ax=axes[1], linestyle='-',c='b',ylabel=r'$\tau_z[N]$',title='A0',label='HM-Optimized')
graph.figureplot(Data['Freq[Hz]'],abs(t_z_a0_mean),ax=axes[1], linestyle='--',marker='*',c='r',ylabel=r'$\tau_r[N]$', title='A0',label='HM-Optimized-mean', alpha=1,markersize=2)
# graph.figureplot(PF_model._equations.Freq, abs(pf_tA[:,0]),ax=axes[1], linestyle='-',c='k',ylabel=r'$\tau_r[N]$', title='A0',label='PF')
graph.figureplot(Data['Freq[Hz]'], abs(t11_A0),ax=axes[1], linestyle='None',marker='o',c='k',ylabel=r'$\tau_z[N]$', title='A0',label='FEM', path=pathsave,filename='tZZ', alpha=1,markersize=2)
isdataSave=True

if isdataSave:
    stress_ZZ_opt={}
    pathOpt="K:\LMC\Sanjay\Code\Optimization\optimization_stress\Optimized_stress_const_RR\\"
    stress_ZZ_opt['Freq[Hz]']=Data['Freq[Hz]']
    stress_ZZ_opt['tzS0']=(t_z_s0)
    stress_ZZ_opt['tzS0_mean']=(t_z_s0_mean)
    stress_ZZ_opt['tzA0']=(t_z_a0)
    stress_ZZ_opt['tzA0_mean']=(t_z_a0_mean)
    df=pd.DataFrame.from_dict(stress_ZZ_opt)
    df.to_csv(pathOpt+'stress_ZZ_optimised.csv')

plt.show()

