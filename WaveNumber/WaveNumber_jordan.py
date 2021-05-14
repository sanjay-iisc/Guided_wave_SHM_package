import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.matlab import loadmat
import numpy as np
import Importing_civaWavenumber as Im
from scipy.interpolate import interp1d
# plt.style.use(['science','ieee', 'grid'])
def possitiveValues(X):
    Len=10
    Matrix=[]
    for j in range(Len):
        P, N= [],[]
        for i, value in enumerate(X[:,j]):
            if value >= 0:
                P.append(value)
            else:
                P.append(0)
        Matrix.append(P)
    return Matrix 

def Nagative_Values(X):
    Len=10
    Matrix=[]
    for j in range(Len):
        P, N= [],[]
        for i, value in enumerate(X[:,j]):
            if value < 0:
                N.append(value)
            else:
                N.append(0)
        Matrix.append(N)
    return Matrix 

def fillterning_List(X): 
    Length=len(X)
    temp =0
    for i in range(Length):
        Array=X[i]
        for j in np.arange(i+1,Length,1):
            tempArray= X[j]
            if len(Array) > len(tempArray):
                m = len(tempArray)
            else:
                m = len(Array)
        
            for kk in range(m):
                if Array[kk]==0:
                    X[i][kk]=tempArray[kk]
                    X[j][kk]=0
                    sanjay=tempArray[kk]
                elif Array[kk]==tempArray[kk]:
                    X[j][kk]=0
            
    return X

#%%                 
mat = scipy.io.loadmat('E:\Work\Code\matlabJordan\calcul_modal\\mesh_size_1_3\\dispersion.mat')
h=1.5 ## in mm
Freq=(mat['fh'].T/h).flatten()
#%%
matReal = pd.read_csv('E:\Work\Code\matlabJordan\calcul_modal\\mesh_size_100\\All_modes\\Real_waveNumber.csv', header=None)
matImag = pd.read_csv('E:\Work\Code\matlabJordan\calcul_modal\\mesh_size_100\\All_modes\\Imag_waveNumber.csv', header=None)
ArrayReal=matReal.to_numpy()
ArrayImag=matImag.to_numpy()
g, W1 = Im.CivaData("E:\Work\Work\CivaResult\Aluminum_5086","Nicolas_waveNumber15mm.txt") 
P=possitiveValues(ArrayReal)
N=Nagative_Values(ArrayImag)
M=fillterning_List(P)
T=fillterning_List(N)
filterComp_wave=np.array(M).T+1j*np.array(N).T
Comp_wave=ArrayReal+1j*ArrayImag
K = mat['Wavenumber']
Keven = mat['Wavenumber_eva']
K = mat['Wavenumber']
Keven = mat['Wavenumber_eva']

Wavenumber=np.concatenate((K,1j*Keven[:,:20],Comp_wave[:,:50]), axis=1)
#%%%
data = loadmat('K:\LMC\Sanjay\ForOlivier\\NicolasCode\\CheckWavenumber\\20210420_1255_alu_1_5mm_Sanjay_400el.mat')
kb = data['z'][:,0]*1e3
fb = data['z'][:,1]
plt.figure()
for i in np.arange(1,10,1):
    plt.scatter(Freq,np.real(Wavenumber[:,i]*1e3), s=1, )
    plt.scatter(Freq,np.imag(Wavenumber[:,i])*1e3, s=1, c='k')
plt.scatter(fb, np.real(kb), marker='*', c='r')
plt.scatter(fb, np.imag(kb), marker='*', c='b')
plt.plot(W1,g[:,3]*1e3, linewidth=5, alpha=0.5, label = 'CIVA-A0')
plt.plot(W1,g[:,0]*1e3, linewidth=5, alpha=0.5, label = 'CIVA-S0')
plt.xlabel('Frequency [MHz]')
plt.ylabel('K[rad/m]')
plt.title('WaveNumber curve for 1,5 mm plate')
# np.save("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix_100mesh.npy",Wavenumber)
# np.save("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix_100mesh.npy",Freq)
plt.show()

    