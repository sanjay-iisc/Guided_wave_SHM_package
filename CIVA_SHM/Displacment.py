import sys
sys.path.append("./")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GuidedWaveModelling.Figure_plot as graph
#-------------import the comsol
pathcomsol="K:\LMC\Sanjay\Comsolresults\\NicolasResults\TimeDomain\\comsol_125mm_u_25mm_upper.csv"
DispFEM=pd.read_csv(pathcomsol, skiprows=4)



#------------
def Displacement_CIVA(path):
    Disp=pd.read_csv(path+'Sensor_1.txt',skiprows=1,sep=' ',header=0)#nrows=3
    T,U,V,W=Disp.iloc[:,0],Disp.iloc[:,1::3],Disp.iloc[:,2::3],Disp.iloc[:,3::3]
    N =U.shape[1]
    U_up,V_up,W_up=U.iloc[:,:N//2].to_numpy(),V.iloc[:,:N//2].to_numpy(),W.iloc[:,:N//2].to_numpy()
    U_down,V_down,W_down=U.iloc[:,N//2:].to_numpy(),V.iloc[:,N//2:].to_numpy(),W.iloc[:,N//2:].to_numpy()
    Vs=(V_up-V_down)*0.5
    Va=(V_up+V_down)*0.5
    return T, V_up[:,5]

path_PF="E:\Work\Work\\Nicolas_Simulation\\configration_500Kh_PF_effectiveRadius.civa\\proc0\\result\Displacement\\"
path_PF_EFradius="E:\Work\Work\\Nicolas_Simulation\\configration_500Kh_PF.civa\\proc0\\result\Displacement\\"

path_HM="E:\Work\Work\\Nicolas_Simulation\\configration_500Kh_2.civa\\proc0\\result\Displacement\\"
TPF,UrPF=Displacement_CIVA(path_PF)
THM,UrHM=Displacement_CIVA(path_HM)

fig,axes=plt.subplots()
graph.figureplot(TPF,UrPF*1e-3*0.06277162055852635,ax=axes ,label='CIVA-PF', title=r'$U_r$'+' @25mm',c='r')
graph.figureplot(THM,UrHM*1e-3,ax=axes, label='HM',c='k')
graph.figureplot(DispFEM['% Time (s)']*1e6,DispFEM['Displacement field, R component (mm), Point: (prob, plateThickness)'].str.replace('i','j').apply(lambda x: np.complex128(x))
, label='Comsol',ax=axes, title=r'$U_r$'+' @25mm', marker='o',markersize=2, linestyle='--', c='b')
plt.xlim([0,30])

plt.show()


