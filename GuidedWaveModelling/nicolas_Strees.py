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
    print(S1)
F=np.arange(5, 1000, 20)*1e3
plt.figure()
plt.plot(F,tr)
plt.plot(F,fAw(F))

Ks=interp1d(x,Ka)(F)
x2=1.11
x3=0.93
x4=-0.22
t_r=np.array(tr)*fAw(F)*jv(x2,Ks*a*x3)*(Ks*a)**x4
plt.figure()
plt.plot(F, abs(t_r))

plt.show()