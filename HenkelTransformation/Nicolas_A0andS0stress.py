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
# from scipy.interpolate import interp1d
from scipy import interpolate
#%%
def stressRZ(p):
    path='K:\\LMC\\Sanjay\\Comsolresults\\NicolasResults\\stressnew2' 
    fileName= 'Stress_RZ_'+str(p)+'.csv' # the file is starting from 1
    df = pd.read_csv(os.path.join(path , fileName), skiprows=8)
    ##
    columns =df.columns
    df.columns=['Radius (mm)', 'thickness (mm)', 'stress (N/m^2)', 'f0']
    SigmaRz= df['stress (N/m^2)'].str.replace('i','j').apply(lambda x: np.complex128(x))*1e-6
    R=df['Radius (mm)']
    return R, SigmaRz
def stressZZ(p):
    path='K:\\LMC\\Sanjay\\Comsolresults\\NicolasResults\\stressnew2' 
    fileName= 'Stress_ZZ_'+str(p)+'.csv' # the file is starting from 1
    df = pd.read_csv(os.path.join(path , fileName), skiprows=8)
    ##
    columns =df.columns
    df.columns=['Radius (mm)', 'thickness (mm)', 'stress (N/m^2)', 'f0']
    SigmaZz= df['stress (N/m^2)'].str.replace('i','j').apply(lambda x: np.complex128(x))*1e-6
    R=df['Radius (mm)']
    return R, SigmaZz

def waveNumber_nicolasPlate(Freq_comp,Mode=1):
    K = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix.npy")#rad/mm ##in rad mm
    Freq_in = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix.npy")*1e3 ## in khz
    f = interp1d(Freq_in,K[:,Mode]) ## interpolating the function for getting the specific k
    return abs(f(Freq_comp))


def wavenumber_stressRZ(p,k):
    r = np.linspace(0,7,1000) # interpolating at new x-axis
    fnew=funStressRZ(r,p,isPlotting=False) # interpolating the function and make the zero at the outer side of the function 
    k,Fk=henkelTransformation(fnew,k,NU=1)
    return Fk
def wavenumber_stressZZ(p,k):
    r = np.linspace(0,7,1000) # interpolating at new x-axis
    fnew=funStressZZ(r,p,isPlotting=False) # interpolating the function and make the zero at the outer side of the function 
    k,Fk=henkelTransformation(fnew,k,NU=0)
    return Fk

def funStressRZ(r,p, isPlotting=True):
    x,y=stressRZ(p)
    # fun= interp1d(x,y, kind='cubic')
    fnew=interp1d(x,real(y),fill_value=(0, 0) ,bounds_error=False)#Spline(x,real(y), k=1)

    if isPlotting:
        plt.figure()
        plt.plot(r, fnew(r), linewidth=3, alpha=0.3)
        # plt.plot(x, real(y), marker='o')
    return fnew

def funStressZZ(r,p, isPlotting=True):
    x,y=stressZZ(p)
    # fun= interp1d(x,y, kind='cubic')
    fnew=interp1d(x,real(y),fill_value=(0, 0) ,bounds_error=False)#Spline(x,real(y), k=1)

    if isPlotting:
        plt.figure()
        plt.plot(r, fnew(r), linewidth=3, alpha=0.3)
        # plt.plot(x, real(y), marker='o')
    return fnew

def henkelTransformation(f,k,NU=1):
    ht = HankelTransform(    nu= NU,     # The order of the bessel function
    N = 1024*4,   # Number of steps in the integration
    h = 0.000001)   # Proxy for "size" of steps in integration
    # k = 0.4#np.logspace(-4,1,400)               # Create a log-spaced array of k from 0.1 to 10.
    Fk = ht.transform(f,k,ret_err=False) # Return the transform of f at k.  
    return k,Fk
###https://stackoverflow.com/questions/43259276/python-fast-hankel-transform-for-1d-array

def A0_stress(isPlotting=True):
   
    P = np.arange(1, 200,1) ## filenumber
    Freq = np.arange(5, 1000, 5) ## in Khz
    
    stress_AR=[]
    stress_AZ=[]
    for i,p in enumerate(P):
        k=waveNumber_nicolasPlate(Freq[i],Mode=1)
        stress_AR.append(wavenumber_stressRZ(p,k))
        stress_AZ.append(wavenumber_stressZZ(p,k))
    return np.array(stress_AR),np.array(stress_AZ)

def S0_stress(isPlotting=True):
   
    P = np.arange(1, 200,1) ## filenumber
    Freq = np.arange(5, 1000, 5) ## in Khz
    
    stress_SR=[]#np.zeros_like(P)
    stress_SZ=[]
    for i,p in enumerate(P):
        k=waveNumber_nicolasPlate(Freq[i],Mode=2)
        stress_SR.append(wavenumber_stressRZ(p,k))
        stress_SZ.append(wavenumber_stressZZ(p,k))
    return np.array(stress_SR),np.array(stress_SZ)
   
    
#%%        
if __name__=='__main__':

    stress_AR,stress_AZ=A0_stress()
    stress_SR,stress_SZ=S0_stress()
    Freq = np.arange(5, 1000, 5)
    # %%
    fig, axes=plt.subplots(1,2)
    fig.axes[0].plot(Freq,abs(np.array(stress_AR)), linestyle='None', marker='o' )
    fig.axes[1].plot(Freq,abs(np.array(stress_SR)), linestyle='None', marker='o' )
    fig, axes=plt.subplots(1,2)
    fig.axes[0].plot(Freq,abs(np.array(stress_AZ)), linestyle='None', marker='o' )
    fig.axes[1].plot(Freq,abs(np.array(stress_SZ)), linestyle='None', marker='o' )
    # %%
    np.save("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAR.npy",stress_AR)
    np.save("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAZ.npy",stress_AZ)
    np.save("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSR.npy",stress_SR)
    np.save("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSZ.npy",stress_SZ)
    plt.show()
    # p=1
    # k,Fk=wavenumber_stress(p)
    # plt.figure()
    # plt.scatter(k,Fk)
    # plt.show()
    # x,t=stressRZ(p)
    # r = np.linspace(0,6,1000)
    # fnew=funStress(r,p,isPlotting=True)
    # k,Fk=henkelTransformation(fnew)
    # fig, axes=plt.subplots(2,1)
    # fig.axes[0].plot(k, (Fk))
    # fig.axes[1].plot(x, real(t))
    # fig.axes[0].set_ylim([-max(real(t)),max(real(t))+max(real(t))*0.5])
    # plt.show()

    
# %%
