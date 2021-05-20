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

data={}
##---Import the data from the optimization
# path="K:\LMC\Sanjay\Code\Optimization\\optimization_stress\ZZ\\"
Freq=np.concatenate((np.arange(5,200,10),np.arange(200,500,10),np.arange(500,1000,10)))
X0,X1,X2,X3,X4=[],[],[],[],[]
#--------------------------------------------------
for nfreq in Freq:
    path="K:\LMC\Sanjay\Code\Optimization\\optimization_stress\RR\\"+'F_'+str(nfreq)+'_KHz'
    fileName='Best_solution.csv'
    df= pd.read_csv(os.path.join(path,fileName))
    X0.append(float(df['X0']))
    X1.append(float(df['X1']))
    X2.append(float(df['X2']))
    X3.append(float(df['X3']))
    # X4.append(df['X4'])
#---- coverting the data in array 
X1=np.array(X1)
X2=np.array(X2)
X3=np.array(X3)
X0=np.array(X0)

#-------Import data----------------------------------------------
data['Freq[Hz]']=Freq*1e3
data['tr']=X0
data['alpha']=X1
data['gamma']=X2
data['beta']=X3
data['A(w)']=fAw(data['Freq[Hz]'])
#------------Data To save----------------------------------
Data=pd.DataFrame.from_dict(data)
# Data.to_csv("K:\LMC\Sanjay\Code\Optimization\optimization_stress\Optimized_stress_const_RR\stress_hyperperameter_RR.csv")
Data_mean=Data.mean(axis=0)
print(Data_mean)
KS0=interp1d(freqModel,Ks)(data['Freq[Hz]'])
KA0=interp1d(freqModel,Ka)(data['Freq[Hz]'])

t_r_s0=data['tr']*data['A(w)']*jv(data['alpha'],KS0*data['gamma']*a)*(KS0*a)**data['beta']
#
# t_r_s0=data['tr']*data['A(w)']*jv(1,KS0*data['gamma']*a)
t_r_s0_mean=Data['tr']*data['A(w)']*jv(Data_mean['alpha'],KS0*Data_mean['gamma']*a)*(KS0*a)**Data_mean['beta']

t_r_a0=data['tr']*data['A(w)']*jv(data['alpha'],KA0*data['gamma']*a)*(KA0*a)**data['beta']

t_r_a0_mean=data['tr']*fAw(data['Freq[Hz]'])*jv(Data_mean['alpha'],KA0*Data_mean['gamma']*a)*(KA0*a)**Data_mean['beta']
# t_r_a0_mean=data['tr']*data['A(w)']*jv(1.11,KA0*0.92*a)*(KA0*a)**-0.22
#-----PF Model
PF_model=PF.Displacement_Field_PF()  
UPF,pf_tS,pf_tA=PF_model.PF_Displacement(isPlotting=False)

#--------------FEM -Stress
FemFreq = np.arange(5, 1000, 5)*1e3
t11_S0=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\\calcul_modal\\NicolasPlate\stressSR.npy"))(data['Freq[Hz]'])
t11_A0=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\\calcul_modal\\NicolasPlate\stressAR.npy"))(data['Freq[Hz]'])

path ="E:\\Work\Work\\Nicolas_opti_results\\"
tr=[]
for nFreq in np.arange(5, 1000, 20):
    filnameFoldar='F_'+str(int(nFreq))+'_KHz'
    filname='Best_solution.csv'
    data=pd.read_csv(path+filnameFoldar+'\\'+filname)
    
    S1=float(data['BestVar'].str.split("[")[0][1].split("]")[0])
    tr.append(S1)

fig,axes=plt.subplots(1,1)
graph.figureplot(Freq*1e3,(X1),ax=axes, linestyle='--',marker='*',title=r'$\alpha_r$')

fig,axes=plt.subplots(1,1)
graph.figureplot(Freq*1e3,(X2),ax=axes, linestyle='--',marker='*',title=r'$\gamma_r$')

fig,axes=plt.subplots(1,1)
graph.figureplot(Freq*1e3,(X3),ax=axes, linestyle='--',marker='*',title=r'$\beta_r$')


fig,axes=plt.subplots(1,1)
# graph.figureplot(Freq*1e3,(X0),ax=axes, linestyle='--',marker='*',title=r'$\tau_r$')
graph.figureplot(Freq*1e3,(X0),ax=axes, linestyle='--',marker='*',title=r'$\tau_r$')
graph.figureplot(np.arange(5, 1000, 20)*1e3,(tr),ax=axes, linestyle='--',marker='*',title=r'$\tau_r$', label='Era')

fig,axes=plt.subplots(1,2, sharey=True)
#-------------S0------------------------
graph.figureplot(Freq*1e3,abs(t_r_s0),ax=axes[0], linestyle='-',c='b',ylabel=r'$\tau_r[N-mm^2]$',title='S0', label='HM-Optimized')
graph.figureplot(Freq*1e3,abs(t_r_s0_mean),ax=axes[0], linestyle='--',marker='*',c='r',ylabel=r'$\tau_r[N-mm^2]$', title='S0',label='HM-Optimized-mean')
graph.figureplot(PF_model._equations.Freq, abs(pf_tS[:,0]),ax=axes[0], linestyle='-',c='k',ylabel=r'$\tau_r[N-mm^2]$', title='S0',label='PF')
graph.figureplot(Freq*1e3, abs(t11_S0),ax=axes[0], linestyle='None',marker='*',c='k',ylabel=r'$\tau_r[N-mm^2]$', title='S0',label='FEM')
#-----A0------------------------------------------
graph.figureplot(Freq*1e3,abs(t_r_a0),ax=axes[1], linestyle='-',c='b',ylabel=r'$\tau_r[N-mm^2]$',title='A0',label='HM-Optimized')
graph.figureplot(Freq*1e3,abs(t_r_a0_mean),ax=axes[1], linestyle='--',marker='*',c='r',ylabel=r'$\tau_r[N-mm^2]$', title='A0',label='HM-Optimized-mean')
graph.figureplot(PF_model._equations.Freq, abs(pf_tA[:,0]),ax=axes[1], linestyle='-',c='k',ylabel=r'$\tau_r[N-mm^2]$', title='A0',label='PF')
graph.figureplot(Freq*1e3, abs(t11_A0),ax=axes[1], linestyle='None',marker='*',c='k',ylabel=r'$\tau_r[N-mm^2]$', title='A0',label='FEM')

path ="E:\\Work\Work\\Nicolas_opti_results\\"




plt.show()

