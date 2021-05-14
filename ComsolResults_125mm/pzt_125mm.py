import sys
sys.path.append("./")
import numpy as np
# import pandas as pd
from numpy import real, imag
import matplotlib.pyplot as plt
from GuidedWaveModelling import newGuidedwavePropagation as Analytical
from scipy.interpolate import interp1d
dis = pd.read_csv("K:\LMC\Sanjay\Comsolresults\\NicolasResults\\NicolasResults5.csv", skiprows=4)

bondedfreq=dis['freq (kHz)'].to_numpy()*1e3 #in Hz

UtipFEm=dis['Total displacement (mm), Point: (5, 1.5)']
UtipFEmR=dis['Displacement field, R component (mm), Point: (5, 1.5)'].str.replace('i','j').apply(lambda x: np.complex128(x))
UtipFEmZ=dis['Displacement field, Z component (mm), Point: (5, 1.5)'].str.replace('i','j').apply(lambda x: np.complex128(x))
absUtip= np.sqrt(real(UtipFEmR)**2+real(UtipFEmZ)**2)
#################---------------->>>>>
Wave =Analytical.WaveDisplacment()
Wave.a=5
Wave.alpha_r =1.11#1.11
Wave.beta_r = -0.22
Wave.alpha_z =1.17
Wave.beta_z = 0.41
Wave.zeta =0.13
Wave.shearLeg_r =0.93
Wave.shearLeg_z = 0.92
Wave.hp=0.125
UtipAna=Wave.PRHW(isPlotting=False, isSavefig=False)

#####
NormaLTip=(UtipAna)#/(20*26.4)
Ffem=interp1d(bondedfreq,absUtip)
Fana=interp1d(Wave.Freq*1e6,NormaLTip)
newFreq=np.arange(0.01,0.995,0.01)*1e6

#%%Ploting the figure
plt.figure(1)
# plt.plot(bondedfreq, UtipFEm, '-', c='r', label='FEM-abs')
plt.plot(bondedfreq, absUtip, '-', c='b',label='AN-abs', linewidth=5, alpha=0.5)
plt.plot(Wave.Freq*1e6, (UtipAna)/(20*26.4))
plt.legend()
plt.figure(2)
plt.plot(newFreq*1e6, Ffem(newFreq)/Fana(newFreq), '-', c='b',label='AN-abs', linewidth=5, alpha=0.5)
plt.show()
# %%
