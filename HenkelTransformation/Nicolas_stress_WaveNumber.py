#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scipybessel
import pandas as pd
import hankel
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from hankel import HankelTransform     # Import the basic class
print("Using hankel v{}".format(hankel.__version__))
import os
from numpy import real , imag
from scipy.interpolate import interp1d
import pandas as pd
# from scipy.interpolate import interp1d
from scipy import interpolate
# plt.style.use('./Plotting_style/science.mplstyle')
#%% Importing the stress RZ and ZZ from the comsol
# p is strat from the 1 
def stressRZ(p):
    path='K:\\LMC\\Sanjay\\Comsolresults\\NicolasResults\\stressnew2' 
    fileName= 'Stress_RZ_'+str(p)+'.csv' # the file is starting from 1
    df = pd.read_csv(os.path.join(path , fileName), skiprows=8)
    ##
    columns =df.columns
    df.columns=['Radius (mm)', 'thickness (mm)', 'stress (N/m^2)', 'f0']
    SigmaRz= df['stress (N/m^2)'].str.replace('i','j').apply(lambda x: np.complex128(x))*1e-6# N/mm2
    R=df['Radius (mm)']
    return R, real(SigmaRz)
def stressZZ(p):
    path='K:\\LMC\\Sanjay\\Comsolresults\\NicolasResults\\stressnew2' 
    fileName= 'Stress_ZZ_'+str(p)+'.csv' # the file is starting from 1
    df = pd.read_csv(os.path.join(path , fileName), skiprows=8)
    ##
    columns =df.columns
    df.columns=['Radius (mm)', 'thickness (mm)', 'stress (N/m^2)', 'f0']
    SigmaZz= df['stress (N/m^2)'].str.replace('i','j').apply(lambda x: np.complex128(x))*1e-6
    R=df['Radius (mm)']
    return R, real(SigmaZz)
#%% Defining the Henkel transformation 
# ###https://stackoverflow.com/questions/43259276/python-fast-hankel-transform-for-1d-array
def henkelTransformationRZ(f,NU=1):
    # f is function 
    # k is wave number wector
    # Nu is order of the bessel function.
    H=1e-6
    ht = HankelTransform(    nu= NU,     # The order of the bessel function
    N = 1024*3,   # Number of steps in the integration
    h = H)   # Proxy for "size" of steps in integration
    k = np.logspace(-3,1,400)               # Create a log-spaced array of k from 0.1 to 10.
    Fk = ht.transform(f,k,ret_err=False) # Return the transform of f at k.  
    return k,Fk
def henkelTransformationZZ(f,NU=0):
    # f is function 
    # k is wave number wector
    # Nu is order of the bessel function.
    ht = HankelTransform(    nu= NU,     # The order of the bessel function
    N = 1024*8,   # Number of steps in the integration
    h = 1e-7)   # Proxy for "size" of steps in integration
    k = np.logspace(-2,1,400)               # Create a log-spaced array of k from 0.1 to 10.
    Fk = ht.transform(f,k,ret_err=False) # Return the transform of f at k.  
    return k,Fk
def inverse_henkelTransformationZZ(f,NU=0):
    # f is function 
    # k is wave number wector
    # Nu is order of the bessel function.
    ht = HankelTransform(    nu= NU,     # The order of the bessel function
    N = 1024*8,   # Number of steps in the integration
    h = 1e-7, inveres=True)   # Proxy for "size" of steps in integration
    k = np.logspace(-2,1,400)               # Create a log-spaced array of k from 0.1 to 10.
    Fk = ht.transform(f,k,ret_err=False) # Return the transform of f at k.  
    return k,Fk


#%%%
def funStressRZ(r,p, isPlotting=False):
    x,y=stressRZ(p)
    Freq = np.arange(5, 1000, 5)
    # fun= interp1d(x,y, kind='cubic')
    fnew=interp1d(x,y,fill_value=(0, 0) ,bounds_error=False)#Spline(x,real(y), k=1)

    if isPlotting:
        plt.figure()
        plt.plot(r, fnew(r), linewidth=3, alpha=1, linestyle='None', marker= '*', c='k', label ='Interpolate')
        plt.plot(x, y, linewidth=5, alpha=0.5, linestyle='-', c='r', label='Original')
        plt.legend()
        plt.xlabel('Radius[mm]')
        plt.ylabel(r'$\sigma_{rz}$'+r'[$N/mm^2$]', fontsize=12)
        plt.title('F='+str(Freq[p-1])+' [KHz]')
        # plt.plot(x, real(y), marker='o')
    return fnew
def funStressZZ(r,p, isPlotting=False):
    x,y=stressZZ(p)
    Freq = np.arange(5, 1000, 5)
    # fun= interp1d(x,y, kind='cubic')
    fnew=interp1d(x,y,fill_value=(0, 0),kind='cubic' ,bounds_error=False)#Spline(x,real(y), k=1)#

    if isPlotting:
        plt.figure()
        plt.plot(r, fnew(r), linewidth=3, alpha=1, linestyle='None', marker= '*', c='k', label ='Interpolate')
        plt.plot(x, y, linewidth=5, alpha=0.5, linestyle='-', c='r', label='Original')
        plt.legend()
        plt.xlabel('Radius[mm]')
        plt.ylabel(r'$\sigma_{zz}$'+r'[$N/mm^2$]', fontsize=12)
        plt.title('F='+str(Freq[p-1])+' [KHz]')
        # plt.plot(x, real(y), marker='o')
    return fnew

    
#%%        
if __name__=='__main__':
    r = np.linspace(0,7,1000) # interpolating at new x-axis
    fig, axes=plt.subplots(2,2,sharex=False,sharey=True)
    data_max_value=[]
    data_stress_RZ_real={}
    data_stress_ZZ_real={}
    data_stress_RZ_waveNumber={}
    data_stress_ZZ_waveNumber={}
    data_stress_RZ_real['Radius[mm]']=r
    data_stress_ZZ_real['Radius[mm]']=r
    for p in np.arange(10,11,1):
        print('p=',p)
        Freq = np.arange(5, 1000, 5)
        sigma_rz=funStressRZ(r,p, isPlotting=True)
        sigma_zz=funStressZZ(r,p, isPlotting=True)
        data_max_value.append(max(sigma_rz(r)))
        # fig.axes[0].scatter(p,max(sigma_rz(r)),s=1)
        data_stress_RZ_real['sigma_RZ[N/mm^2] '+'F='+str(Freq[p-1])+' [KHz]']=sigma_rz(r)
        data_stress_ZZ_real['sigma_ZZ[N/mm^2] '+'F='+str(Freq[p-1])+' [KHz]']=sigma_zz(r)
        Krz,sigmak_rz=henkelTransformationRZ(sigma_rz,NU=1)
        Kzz,sigmak_zz=henkelTransformationZZ(sigma_zz,NU=0)
        data_stress_RZ_waveNumber['sigma_RZ[N/mm^2] '+'F='+str(Freq[p-1])+' [KHz]']=sigmak_rz
        data_stress_ZZ_waveNumber['sigma_ZZ[N/mm^2] '+'F='+str(Freq[p-1])+' [KHz]']=sigmak_zz
    #     # %%
        
        fig.axes[0].plot(r, sigma_rz(r), label='F='+str(Freq[p-1])+' [KHz]')
        fig.axes[1].plot(Krz,sigmak_rz, linewidth=1, alpha=1, linestyle='-', label='F='+str(Freq[p-1])+' [KHz]')
        fig.axes[2].plot(r, sigma_zz(r), label='F='+str(Freq[p-1])+' [KHz]')
        fig.axes[3].plot(Kzz,sigmak_zz, linewidth=1, alpha=1, linestyle='-', label='F='+str(Freq[p-1])+' [KHz]')
        
        fig.axes[2].set_xlabel('Radius[mm]')
        fig.axes[3].set_xlabel('WaveNumber[rad/mm]')
        fig.axes[1].set_xlim([0,3])
        fig.axes[3].set_xlim([0,3])
        fig.axes[0].set_ylabel(r'Re($\sigma_{rz}$)'+r'[$N/mm^2$]', fontsize=10)
        fig.axes[2].set_ylabel(r'Re($\sigma_{zz}$)'+r'[$N/mm^2$]', fontsize=10)
        # fig.axes[0].title('F='+str(Freq[p-1])+' [KHz]')
        fig.axes[0].legend()
        fig.axes[1].legend()
        fig.axes[2].legend()
        fig.axes[3].legend()
    # # data_stress_RZ_waveNumber['K[rad/mm]']=Krz
    # # data_stress_ZZ_waveNumber['K[rad/mm]']=Kzz
    # # data_stress_RZ_waveNumber=pd.DataFrame.from_dict(data_stress_RZ_waveNumber)
    # # data_stress_ZZ_waveNumber=pd.DataFrame.from_dict(data_stress_ZZ_waveNumber)
    # # data_stress_RZ_real=pd.DataFrame.from_dict(data_stress_RZ_real)
    # # data_stress_ZZ_real=pd.DataFrame.from_dict(data_stress_ZZ_real)
    # plt.savefig('E:\PPT\Presentation\\02052021_ppt\Figure\\stress_wavenumber.png')
    # np.save("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\Maxstress.np", np.array(data_max_value))
    plt.show()
    # data_stress_RZ_waveNumber.to_csv("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\data_stress_RZ_waveNumber.csv")
    # data_stress_ZZ_waveNumber.to_csv("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\data_stress_ZZ_waveNumber.csv")

    # data_stress_RZ_real.to_csv("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\data_stress_RZ_real.csv")
    # data_stress_ZZ_real.to_csv("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\data_stress_ZZ_real.csv")


