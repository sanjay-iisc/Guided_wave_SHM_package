import numpy as np
import scipy.special
import matplotlib.pyplot as plt
plt.style.use('.\Plotting_style\science.mplstyle')
def f(K,r):
    return scipy.special.hankel2(1,K * r) 

K4= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix.npy")*1e3)#rad/mm
Freq4 = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix.npy")*1e6#MHz

# K4=np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrixbastien.npy") 
# Freq4 =np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrixbastien.npy")[:,0] 
Data=[]
r=5e-3#[m]
for i in range(np.shape(K4)[0]):
    x=K4[i,:]
    Data.append(f(x,r))
plt.figure()
for i in np.arange(0,900,150):
    plt.plot(np.arange(0,73,1), abs(Data[i]), linestyle='--', marker='o', label='F[Mhz] :'+str(np.round(Freq4[i]/1e6,4)), markersize=1)
plt.xlabel('$No. modes$')
plt.ylabel(r'$|H^2_{1}(K,r)|$')
plt.title('Convergence of Hankel Function with modes at R=5mm')
plt.legend()

plt.savefig('E:\PPT\Presentation\\04052021_ppt\Figure\\converganceHankel.png')
plt.show()