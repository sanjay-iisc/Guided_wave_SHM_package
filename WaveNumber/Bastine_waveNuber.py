from scipy.io.matlab import loadmat
import matplotlib.pyplot as plt
import numpy as np
data = loadmat('K:\LMC\Sanjay\ForOlivier\\NicolasCode\\CheckWavenumber\\20210420_1255_alu_1_5mm_Sanjay_400el.mat')
Kb = data['z'][:,0]*1e3
Fb = data['z'][:,1]*1e6
# freq=np.arange(0.01,1+0.01,0.01)
a=[]#np.zeros((102,80), dtype=complex)
count=0
index=0
a= Kb.reshape((100,70))
freq=abs(Fb).reshape((100,70))
    
#     # a[count][count]=(Kb[np.argmin(np.abs(f-Fb[n::69]))])
#     # count=+1    

np.save("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrixbastien.npy",a)
np.save("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrixbastien.npy",freq)
# a= np.array(a, dtype=complex)
print(a)
plt.figure()
for n in range(1):
    plt.scatter(freq[:,0], a[:,n])
    # plt.scatter(freq, np.imag(a))
plt.show()