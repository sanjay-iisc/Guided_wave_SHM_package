from os import name
import sys
sys.path.append("./")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GuidedWaveModelling.Figure_plot as graph
from scipy.signal import hilbert
#-------------import the comsol
# pathcomsol="K:\LMC\Sanjay\\Comsolresults\\NicolasResults\\TimeDomain\Results_125mm\\Displacments_at25mm_0.06_Mhz.csv"
# pathcomsol_lineLoad="K:\LMC\Sanjay\\Comsolresults\\NicolasResults\\TimeDomain\Results_125mm\\Displacments_at25mm_0.06_Mhz_lineload.csv"
# DispFEM_lineLoad=pd.read_csv(pathcomsol_lineLoad, skiprows=4)

class FEM:
    def __init__(self, path):
        # path is an 
         self.path=path
         self.U=0
    def setValue(self, x):
        self.U=x

    def getValue(self,index):
        print(index)
        name=self.U.keys()[index]
        print(name)
        return self.U[name]

    def Displacement(self,N_Rows=4):
        print("Importing the FEM-File From {}".format(self.path))
        DispFEM=pd.read_csv(self.path, skiprows=N_Rows)
        print("Column Name in File {}".format(DispFEM.keys()))
        print("Now you can chose the Index to be import by the getValue called")
        self.setValue(DispFEM)
    
    def HT(self,U):
        # Given U first would be the Time 
        self.U_HT_amp=[] ### Hilbert T
        self.U_HT_frequency=[] ### Hilbert T 
        T=U[0]# Time data
        Fs= 1/ (T[1]-T[0])
        for i in np.arange(1, len(U),1):
            analytic_signal = hilbert(U[i])
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * Fs)
            self.U_HT_amp.append(amplitude_envelope)
            self.U_HT_frequency.append(instantaneous_frequency)


class CIVA_simulation:
    def __init__(self, path, index=5):
        # path is an 
         self.path=path
         self.index=index
    def Displacement(self):
        print("Importing the CIVA-File From {}".format(self.path))
        Disp=pd.read_csv(self.path+'Sensor_1.txt',skiprows=1,sep=' ',header=0)#nrows=3
        T,U,V,W=Disp.iloc[:,0],Disp.iloc[:,1::3],Disp.iloc[:,2::3],Disp.iloc[:,3::3]
        N =U.shape[1]
        U_up,V_up,W_up=U.iloc[:,:N//2].to_numpy(),V.iloc[:,:N//2].to_numpy(),W.iloc[:,:N//2].to_numpy()
        U_down,V_down,W_down=U.iloc[:,N//2:].to_numpy(),V.iloc[:,N//2:].to_numpy(),W.iloc[:,N//2:].to_numpy()
        Vs=(V_up-V_down)*0.5
        Va=(V_up+V_down)*0.5
        Ws=(W_up-W_down)*0.5
        Wa=(W_up+W_down)*0.5
        return [T, V_up[:,self.index],W_up[:,self.index],V_down[:,self.index],W_down[:,self.index]]

    def HT(self):
        U=self.Displacement()
        self.U_HT_amp=[] ### Hilbert T
        self.U_HT_frequency=[] ### Hilbert T 
        T=U[0]# Time data
        Fs= 1/ (T[1]-T[0])
        print("***********************************\n")
        print("Sampling Frequency In CIVA {} Mhz".format(Fs))
        print("***********************************\n")
        for i in np.arange(1, len(U),1):
            analytic_signal = hilbert(U[i])
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * Fs)
            self.U_HT_amp.append(amplitude_envelope)
            self.U_HT_frequency.append(instantaneous_frequency)
        
if __name__ =='__main__':
    Cs1=CIVA_simulation("E:\Work\Work\\Nicolas_Simulation\\_30052021_60khz_PF_imput_stress.civa\\proc0\\result\Displacement\\")
    U=Cs1.Displacement()
    Comsol1=FEM("K:\LMC\Sanjay\\Comsolresults\\NicolasResults\\TimeDomain\Results_125mm\\Displacments_at25mm_0.3_Mhz.csv")
    UFEM=Comsol1.Displacement()
    print(Comsol1.getValue(1))





