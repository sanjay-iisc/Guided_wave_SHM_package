#%%
from scipy.io.matlab import loadmat
import matplotlib.pyplot as plt
import numpy as np
#%%
def dads_equation(K,f):
    C_L = 6197.824298019837# * 1e-3
    C_T = 3121.9527052723133# * 1e-3
    W=f*1e6*2*np.pi #* 1e-6
    D= 1.5/2 * 1e-3 #* 1e3
    
    p = np.lib.scimath.sqrt((W / C_L) ** 2 - K ** 2)
    q = np.lib.scimath.sqrt((W / C_T) ** 2 - K ** 2)
    Da1 = np.cos(p * D) * np.sin(q * D) * 4 * (K ** 2) * p * q
    Ds1 = np.sin(p * D) * np.cos(q * D) * 4 * (K ** 2) * p * q
    Da2 = np.cos(q * D) * np.sin(p * D) * (q ** 2 - K ** 2) ** 2
    Ds2 = np.sin(q * D) * np.cos(p * D) * (q ** 2 - K ** 2) ** 2
    return Da1+Da2, Ds2+Ds1
#%%
plt.close('all')
# data = loadmat('C:/Users/om250922/Desktop/testSanjay/20210420_1258_alu_1_5mm_Sanjay_10el.mat')
# data = loadmat('C:/Users/om250922/Desktop/testSanjay/20210420_1254_alu_1_5mm_Sanjay_200el.mat')
data = loadmat('K:\LMC\Sanjay\ForOlivier\\NicolasCode\\CheckWavenumber\\20210420_1255_alu_1_5mm_Sanjay_400el.mat')

ks =  np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix.npy")*1e3
fs =np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix.npy")#np.linspace(0.05,1.005, 100)

kb = data['z'][:,0]*1e3
fb = data['z'][:,1]

#%%
plt.figure()
plt.plot(fs, np.real(ks), 'ok', label = 'Sanjay')
plt.plot(np.abs(fb), np.real(kb), 'xr', label = 'Bastien')
plt.title('real')
plt.xlabel('f [MHz]')
plt.ylabel('real')
# plt.legend()

plt.figure()
plt.plot(fs, np.imag(ks), 'ok', label = 'Sanjay')
plt.plot(np.abs(fb), np.imag(kb), 'xr', label = 'Bastien')
# plt.title('imag')
plt.xlabel('f [MHz]')
plt.ylabel('imag')
# plt.legend()

# single freq slice
f = 0.1
ffs = np.argmin(np.abs(f-fs))
valffb = fb[np.argmin(np.abs(f-fb))]
ffb = fb[:] == valffb

plt.figure()
plt.plot(np.real(ks[ffs, :]), np.imag(ks[ffs, :]), 'ok', label = 'Sanjay')
plt.plot(np.real(kb[ffb]), np.imag(kb[ffb]), 'xr', label = 'Bastien')
plt.title(str(f)+'MHz')
plt.xlabel('real')
plt.ylabel('imag')
plt.legend()


check_kb_a, check_kb_s = dads_equation(np.real(kb[ffb]), f)
check_ks_a, check_ks_s = dads_equation(np.real(ks[ffs, :]), f)
#%%
limy = 30e1
plt.figure()
plt.plot(np.abs(check_ks_a), 'ok', label = 'Sanjay')
plt.plot(np.abs(check_kb_a), 'xr', label = 'Bastien')
plt.title('Check of Da value for all modes at '+str(f)+' MHz')
plt.xlabel('modes')
plt.ylabel('Da(k)')
plt.legend()
plt.ylim((-limy, limy))
# plt.xlim((0, 10))

plt.figure()
plt.plot(np.abs(check_ks_s), 'ok', label = 'Sanjay')
plt.plot(np.abs(check_kb_s), 'xr', label = 'Bastien')
plt.title('Check of Ds value for all modes at '+str(f)+' MHz')
plt.xlabel('modes')
plt.ylabel('Ds(k)')
# plt.ylim((-limy, limy))
# plt.xlim((0, 10))

plt.legend()
plt.show()