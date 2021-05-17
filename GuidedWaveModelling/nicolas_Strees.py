from matplotlib import markers
import numpy as np
import Hybridmodel as HM
import Figure_plot as graph
import matplotlib.pyplot as plt
import scipy.special 
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import os 
import pandas as pd
from scipy.special  import jv
import GuidedWaveModelling.Figure_plot as graph
#---------------------Define the function-------------
#############-----
Model=HM.t_w()
Ks=Model._tipDisp._equations.K[:,2]
Ka=Model._tipDisp._equations.K[:,1]
x=Model._tipDisp._equations.Freq
a=Model._tipDisp._equations.a
# Admittance term
Aw=Model.constan_term(isPlotting=False)
fAw=interp1d(x,Aw)


path ="E:\\Work\Work\\Nicolas_opti_results\\"
tr=[]
for nFreq in np.arange(5, 1000, 20):
    filnameFoldar='F_'+str(int(nFreq))+'_KHz'
    filname='Best_solution.csv'
    data=pd.read_csv(path+filnameFoldar+'\\'+filname)
    
    S1=float(data['BestVar'].str.split("[")[0][1].split("]")[0])
    tr.append(S1)
    # print(S1)
path='E:\Work\Work\\Nicolas_opti_results\ZZ\\'
tz=[]
for nFreq in np.arange(5, 1000, 20):
    filnameFoldar='F_'+str(int(nFreq))+'_KHz'
    filname='Best_solution.csv'
    data=pd.read_csv(path+filnameFoldar+'\\'+filname)
    
    S2=float(data['BestVar'].str.split("[")[0][1].split("]")[0])
    tz.append(S2)
#FEM -Stresss

FemFreq = np.arange(5, 1000, 5)*1e3
t11_S0=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSR.npy"))
t22_S0=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSZ.npy"))

t11_A0=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAR.npy"))
t22_A0=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAZ.npy"))

#---optimize

F=np.arange(5, 1000, 20)*1e3

Ks=interp1d(x,Ks)(F)
Ka=interp1d(x,Ka)(F)
x2=1.11
x3=0.93
x4=-0.22
t_r_S0=np.array(tr)*fAw(F)*jv(x2,Ks*a*x3)*(Ks*a)**x4
t_r_A0=np.array(tr)*fAw(F)*jv(x2,Ka*a*x3)*(Ka*a)**x4

y2=1.17
y3=0.92
y4=0.41
t_z_S0=0.13*np.array(tz)*fAw(F)*jv(y2,Ks*a*y3)*(Ks*a)**y4
t_z_A0=0.13*np.array(tz)*fAw(F)*jv(y2,Ka*a*y3)*(Ka*a)**y4

fig,axes = plt.subplots(1,2, sharey=True)
graph.figureplot(F, abs(t_r_S0),ax=axes[0], title='S0', ylabel=r'$\tau_{rr}$', label ='HM-optimized', linestyle='None', marker='o')
graph.figureplot(F, abs(t11_S0(F)),ax=axes[0], title='S0', ylabel=r'$\tau_{rr}$', label ='HM-FEM')

graph.figureplot(F, abs(t_r_A0),ax=axes[1], title='A0', ylabel=r'$\tau_{rr}$', label ='HM-optimized', linestyle='None', marker='o')
graph.figureplot(F, abs(t11_A0(F)),ax=axes[1], title='A0', ylabel=r'$\tau_{rr}$', label ='HM-FEM')


fig,axes = plt.subplots(1,2, sharey=True)
graph.figureplot(F, abs(t_z_S0),ax=axes[0], title='S0', ylabel=r'$\tau_{zz}$', label ='HM-optimized', linestyle='None', marker='o')
graph.figureplot(F, abs(t22_S0(F)),ax=axes[0], title='S0', ylabel=r'$\tau_{zz}$', label ='HM-FEM')

graph.figureplot(F, abs(t_z_A0),ax=axes[1], title='A0', ylabel=r'$\tau_{zz}$', label ='HM-optimized', linestyle='None', marker='o')
graph.figureplot(F, abs(t22_A0(F)),ax=axes[1], title='A0', ylabel=r'$\tau_{zz}$', label ='HM-FEM')


# plt.figure()
# plt.plot(,label ='HM-optimize', linestyle='None', marker='*')
# plt.plot(F, , label ='FEM')
# plt.xlabel('F[Hz]')
# plt.ylabel()

plt.show()