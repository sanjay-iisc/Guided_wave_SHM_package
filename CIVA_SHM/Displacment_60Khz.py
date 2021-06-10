import numpy as np
import matplotlib.pyplot as plt
from MainFile import *
import pandas as pd
PF_10=CIVA_simulation("E:\Work\Work\\Nicolas_Simulation\\PF\\_02062021_60khz_PF_imput_stress_10dis.civa\\proc0\\result\Displacement\\")
UPF_10=PF_10.Displacement()
PF_10.HT()

CS_10=CIVA_simulation("E:\Work\Work\\Nicolas_Simulation\\_30052021_60khz_PF_input_FEM_stress_10discretized.civa\\proc0\\result\Displacement\\")
U_10=CS_10.Displacement()
CS_10.HT()

CFE=FEM("K:\LMC\Sanjay\\Comsolresults\\NicolasResults\\TimeDomain\Results_125mm\\Displacments_at25mm_0.06_Mhz.csv")
CFE.Displacement()
UFEM=[CFE.getValue(0),np.real(CFE.getValue(1).str.replace('i','j').apply(lambda x: np.complex128(x))),np.real(CFE.getValue(3).str.replace('i','j').apply(lambda x: np.complex128(x))) ]
CFE.HT(UFEM)
#---Line Load
CFE_LL=FEM("K:\LMC\Sanjay\\Comsolresults\\NicolasResults\\TimeDomain\Results_125mm\\Displacments_at25mm_0.06_Mhz_lineLoad.csv")
CFE_LL.Displacement()
UFEM_LL=[CFE_LL.getValue(0),np.real(CFE_LL.getValue(1)),CFE_LL.getValue(2) ]

#---- Reading the Time History
TimeHis=pd.read_csv("E:\Work\Work\\Nicolas_Simulation\CivaFile\\TimeHistory0.06.csv").T
TimeHis.columns=TimeHis.iloc[0]
TimeHis = TimeHis.reset_index(drop=True)
TimeHis=TimeHis.drop([0])

# TimeHis=TimeHis.drop(1)
print(TimeHis)

plt.figure()
plt.plot(UPF_10[0],UPF_10[3]*1e-3, label='CIVA-PF D=10', c='k')
plt.plot(UPF_10[0],PF_10.U_HT_amp[2]*1e-3,c='k', linestyle='--')
plt.plot(U_10[0],U_10[3]*1e-3, label='CIVA-Direct Stress D = 10 ', c='r')
plt.plot(U_10[0],CS_10.U_HT_amp[2]*1e-3,c='r', linestyle='--')
plt.plot(UFEM[0]*1e6,UFEM[1] , label='FEM', c='b', marker='None', markersize=2, linestyle='-')
plt.plot(UFEM[0]*1e6,CFE.U_HT_amp[0],c='b', linestyle='--')

plt.scatter(TimeHis['TS0_Start at XR,YR'],0, marker='o', s=np.pi*10**2,alpha=1, c='g')
plt.scatter(TimeHis['TS0_Complete at XR,YR'],0, marker='o', s=np.pi*10**2,alpha=1, c='r')

plt.scatter(TimeHis['TA0_Start at XR,YR'],0, marker='v', s=np.pi*10**2,alpha=1, c='g')
plt.scatter(TimeHis['TA0_Complete at XR,YR'],0, marker='v', s=np.pi*10**2,alpha=1, c='r')

plt.title('Freq @ 60KHz')
plt.xlabel('Time[micro sec]')
plt.ylabel(r'$U_{rr}[mm]$')
plt.grid()
plt.legend()
path_save="E:\PPT\Presentation\\_01062021_ppt\Figure"
plt.savefig(path_save+'\\'+'Urr60khz.png')
#----------------------------Time Freq---Urr
plt.figure()

plt.plot(U_10[0][1:],CS_10.U_HT_frequency[2],c='r', linestyle='-', label='CIVA')
plt.plot(UFEM[0][1:]*1e6,CFE.U_HT_frequency[0]*1e-6,c='b', linestyle='-', label='FEM')
plt.xlim([4,90])
plt.ylim([0,0.1])
plt.xlabel('Time[micro sec]')
plt.ylabel(r'$Freq[MHZ]$')
plt.title('Freq @ 60KHz- Time Frequency-Urr ')
plt.legend()
plt.grid()
#--------Uzz----------------------------------------
plt.figure()
plt.plot(UPF_10[0],-UPF_10[4]*1e-3, label='CIVA-PF D=10', c='k')
plt.plot(UPF_10[0],PF_10.U_HT_amp[3]*1e-3,c='k', linestyle='--')
plt.plot(U_10[0],-U_10[4]*1e-3, label='CIVA-Direct Stress D = 10 ', c='r',)
plt.plot(U_10[0],CS_10.U_HT_amp[3]*1e-3,c='r', linestyle='--')
plt.plot(UFEM[0]*1e6,UFEM[2] , label='FEM', c='b', marker='None', markersize=2, linestyle='-')
plt.plot(UFEM[0]*1e6,CFE.U_HT_amp[1],c='b', linestyle='--')
# plt.plot(UFEM_LL[0]*1e6,UFEM_LL[2] , label='FEM-LineLoad',c='b', marker='*', markersize=2, linestyle='None')
plt.title('Freq @ 60KHz')
plt.xlabel('Time[micro sec]')
plt.ylabel(r'$U_{zz}[mm]$')
plt.grid()
plt.legend()
plt.scatter(TimeHis['TS0_Start at XR,YR'],0, marker='o', s=np.pi*10**2,alpha=1, c='g')
plt.scatter(TimeHis['TS0_Complete at XR,YR'],0, marker='o', s=np.pi*10**2,alpha=1, c='r')

plt.scatter(TimeHis['TA0_Start at XR,YR'],0, marker='v', s=np.pi*10**2,alpha=1, c='g')
plt.scatter(TimeHis['TA0_Complete at XR,YR'],0, marker='v', s=np.pi*10**2,alpha=1, c='r')
plt.savefig(path_save+'\\'+'Uzz60khz.png')


#----------------------------Time Freq---Uzz
plt.figure()
plt.plot(U_10[0][1:],CS_10.U_HT_frequency[3],c='r', linestyle='-', label='CIVA')
plt.plot(UFEM[0][1:]*1e6,CFE.U_HT_frequency[1]*1e-6,c='b', linestyle='-', label='FEM')
plt.xlim([4,90])
plt.ylim([0,0.1])
plt.xlabel('Time[micro sec]')
plt.ylabel(r'$Freq[MHZ]$')
plt.title('Freq @ 60KHz- Time Frequency-Uzz ')
plt.legend()
plt.grid()



plt.show()