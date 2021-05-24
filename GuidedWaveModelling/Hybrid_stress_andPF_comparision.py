from matplotlib import markers
import numpy as np
import Hybridmodel as HM
import GuidedWaveModelling.PinForce as PF
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
Model=HM.Displacement_Field_FEM()
Freq=Model._equations.Freq
UHM,TS_HMopt,TA_HMopt=Model.Hybrid_Displacement(isPlotting=False)
PF_model=PF.Displacement_Field_PF()  
UPF,pf_tS,pf_tA=PF_model.PF_Displacement(isPlotting=False)
Comsol_Path="K:\LMC\Sanjay\Comsolresults\\NicolasResults\\NicolasResults4.csv"
Data = pd.read_csv(Comsol_Path, skiprows=4)
FemFreq =Data['freq (kHz)'].to_numpy()*1e3 #in Hz
UrS0=Data['S0_u']*1e-3 #in m
UrA0=Data['A0_u']*1e-3#in m
UzS0=Data['S0_w']*1e-3 #in m
UzA0=Data['A0_w']*1e-3#in m

fig,axes=plt.subplots(1,2)
graph.figureplot(Freq,abs(UHM[0]),ax=axes[0], linestyle='None', marker='o', ylabel=r'$U_{rr}[m]$', markersize=3,label='HM-opt',c='b')
graph.figureplot(Freq,abs(UHM[1]),ax=axes[1],linestyle='None', marker='o', ylabel=r'$U_{zz}[m]$', markersize=3,label='HM-opt',c='b')

##-FEM-S0
graph.figureplot(FemFreq,abs(UrS0),ax=axes[0], linestyle='None', marker='o', ylabel=r'$U_{rr}[m]$', markersize=2,label='FEM',c='r')
graph.figureplot(FemFreq,abs(UzS0),ax=axes[1],linestyle='None', marker='o', ylabel=r'$U_{zz}[m]$', markersize=2,label='FEM',c='r')
#----PF
graph.figureplot(PF_model._equations.Freq,abs(UPF[0]),ax=axes[0], linestyle='-', ylabel=r'$U_{rr}[m]$', markersize=1,label='PF',c='k')
graph.figureplot(PF_model._equations.Freq,abs(UPF[1]),ax=axes[1],linestyle='-', ylabel=r'$U_{zz}[m]$', markersize=1,label='PF',c='k')

fig,axes=plt.subplots(1,2)
graph.figureplot(Freq,abs(UHM[2]),ax=axes[0], linestyle='None', marker='o', ylabel=r'$U_{rr}[m]$', markersize=3,label='HM-opt',c='b')
graph.figureplot(Freq,abs(UHM[3]),ax=axes[1],linestyle='None', marker='o', ylabel=r'$U_{zz}[m]$', markersize=3,label='HM-opt',c='b')

##-FEM-A0
graph.figureplot(FemFreq,abs(UrA0),ax=axes[0], linestyle='None', marker='o', ylabel=r'$U_{rr}[m]$', markersize=2,label='FEM',c='r')
graph.figureplot(FemFreq,abs(UzA0),ax=axes[1],linestyle='None', marker='o', ylabel=r'$U_{zz}[m]$', markersize=2,label='FEM',c='r')
#----PF-A0
graph.figureplot(PF_model._equations.Freq,abs(UPF[2]),ax=axes[0], linestyle='-', ylabel=r'$U_{rr}[m]$', markersize=1,label='PF',c='k')
graph.figureplot(PF_model._equations.Freq,abs(UPF[3]),ax=axes[1],linestyle='-', ylabel=r'$U_{zz}[m]$', markersize=1,label='PF',c='k')


plt.show()
 
