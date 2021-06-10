import numpy as np
import matplotlib.pyplot as plt
from MainFile import *
import pandas
path_save="E:\\PPT\\Presentation\\_01062021_ppt\\Figure_400mm\\"
path_FEM="K:\\LMC\\Sanjay\\Comsolresults\\NicolasResults\\TimeDomain\\Results_400mm\\TimeDomainReults\\"
filename_FEM="Displacment_Upper_Freq_0.3_MHz.csv"

path_PF="E:\Work\Work\\Nicolas_Simulation\\PF\_02062021_300khz_PF_imput_stress_4dis.civa\\proc0\\result\Displacement\\"
path_DF="E:\Work\Work\\Nicolas_Simulation\\HM_FEMstress\_04062021_300khz_input_FEM_stress_10discretized.civa\\proc0\\result\Displacement\\"
#------FEM Results---------
Freq=300
Distance=100
Index_CSV=4 # at 100mm
offset_csv_pour_w=4
Index_CIVA=Distance-20
#-----------------------------
CFE=FEM(path_FEM+filename_FEM)
CFE.Displacement()
UFEM=[CFE.getValue(0),np.real(CFE.getValue(Index_CSV).str.replace('i','j').apply(lambda x: np.complex128(x))),
np.real(CFE.getValue(Index_CSV+offset_csv_pour_w).str.replace('i','j').apply(lambda x: np.complex128(x))) ]
CFE.HT(UFEM)
## --- CIVA-RESULTS-FEM 

DS=CIVA_simulation(path_DF, index=Index_CIVA)
U=DS.Displacement()
DS.HT()

#--- CIVA-PIN-FORCE
PF=CIVA_simulation(path_PF,  index=Index_CIVA)
PF_U=PF.Displacement()
PF.HT()
#---- Time_Domain For Mode
TimeHis=pd.read_csv("E:\Work\Work\\Nicolas_Simulation\CivaFile\\TimeHistory0.3.csv").T
TimeHis.columns=TimeHis.iloc[0]
TimeHis = TimeHis.reset_index(drop=True)
TimeHis=TimeHis.drop([0])
###--- Plotting 
##--URrr
plt.figure()

plt.plot(U[0],U[1]*1e-3, label='CIVA-Direct Stress D=4', c='r')
plt.plot(U[0],DS.U_HT_amp[0]*1e-3,c='r', linestyle='--')

plt.plot(PF_U[0],PF_U[1]*1e-3, c='k', linestyle='-', label ='PF D=4')
plt.plot(PF_U[0],PF.U_HT_amp[0]*1e-3,c='k', linestyle='--')

# plt.plot(Hm_U_10[0],Hm_U_10[1]*1e-3/ (5), label='CIVA-HM D=4', c='g')
# plt.plot(Hm_U_10[0],(Hm_CS_10.U_HT_amp[0]*1e-3)/5,c='g', linestyle='--')

plt.plot(UFEM[0]*1e6,UFEM[1] , label='FEM', c='b', marker='None', markersize=2, linestyle='-')
plt.plot(UFEM[0]*1e6,CFE.U_HT_amp[0],c='b', linestyle='--')

plt.title('Freq @ '+str(Freq)+'KHz')
plt.xlabel('Time[micro sec]')
plt.ylabel(r'$U_{rr}[mm]$')
plt.grid()
plt.legend()
x1=float(TimeHis['TA0_Start at XR,YR'])
x2=float(TimeHis['TA0_Complete at XR,YR'])
x= np.linspace(x1,x2,10)
y1=[float(np.min((UFEM[1])))]*10
y2=[float(np.max((UFEM[1])))]*10
# # print(y1)
plt.fill_between(x, y1, y2,
                 facecolor="orange", # The fill color
                 color='blue',       # The outline color
                 alpha=0.2)

x1=float(TimeHis['TS0_Start at XR,YR'])
x2=float(TimeHis['TS0_Complete at XR,YR'])
x= np.linspace(x1,x2,10)
y1=[float(np.min((UFEM[1])))]*10
y2=[float(np.max((UFEM[1])))]*10
plt.fill_between(x, y1, y2,
                 facecolor="orange", # The fill color
                 color='yellow',       # The outline color
                 alpha=0.2)
# plt.xlim([0,50])
plt.savefig(path_save+'\\'+'Urr'+str(Freq)+'_khz'+str(Distance)+'_mm.png')


##--Uzz
plt.figure()

plt.plot(U[0],U[2]*1e-3, label='CIVA-Direct Stress D=4', c='r')
plt.plot(U[0],DS.U_HT_amp[1]*1e-3,c='r', linestyle='--')

plt.plot(PF_U[0],PF_U[2]*1e-3, c='k', linestyle='-', label ='PF D=4')
plt.plot(PF_U[0],PF.U_HT_amp[1]*1e-3,c='k', linestyle='--')

# plt.plot(Hm_U_10[0],Hm_U_10[1]*1e-3/ (5), label='CIVA-HM D=4', c='g')
# plt.plot(Hm_U_10[0],(Hm_CS_10.U_HT_amp[0]*1e-3)/5,c='g', linestyle='--')

plt.plot(UFEM[0]*1e6,UFEM[2] , label='FEM', c='b', marker='None', markersize=2, linestyle='-')
plt.plot(UFEM[0]*1e6,CFE.U_HT_amp[1],c='b', linestyle='--')


plt.title('Freq @ '+str(Freq)+'KHz')
plt.xlabel('Time[micro sec]')
plt.ylabel(r'$U_{zz}[mm]$')
plt.grid()
plt.legend()

x1=float(TimeHis['TA0_Start at XR,YR'])
x2=float(TimeHis['TA0_Complete at XR,YR'])
x= np.linspace(x1,x2,10)
y1=[float(np.min((UFEM[1])))]*10
y2=[float(np.max((UFEM[1])))]*10
# # print(y1)
plt.fill_between(x, y1, y2,
                 facecolor="orange", # The fill color
                 color='blue',       # The outline color
                 alpha=0.2)

x1=float(TimeHis['TS0_Start at XR,YR'])
x2=float(TimeHis['TS0_Complete at XR,YR'])
x= np.linspace(x1,x2,10)
y1=[float(np.min((UFEM[1])))]*10
y2=[float(np.max((UFEM[1])))]*10
plt.fill_between(x, y1, y2,
                 facecolor="orange", # The fill color
                 color='yellow',       # The outline color
                 alpha=0.2)

# plt.xlim([0,50])
plt.savefig(path_save+'\\'+'Uzz'+str(Freq)+'_khz'+str(Distance)+'_mm.png')
plt.show()