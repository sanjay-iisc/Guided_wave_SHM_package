import sys
sys.path.append("./")

import matplotlib.pyplot as plt
import scipy.special 
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
from scipy import integrate
import GuidedWaveModelling.Hybridmodel as HM
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

def funStressRZ(p, isPlotting=False):
    x,y=stressRZ(p)
    Freq = np.arange(5, 1000, 5)
    # fun= interp1d(x,y, kind='cubic')
    fnew=Spline(x,(real(y)), k=1)#interp1d(x,y,fill_value=(0, 0) ,bounds_error=False)#
    Tw=max(np.real(y))#np.sum(fnew(x))/len(x)#integrate.simps(fnew(x), x)#max(np.real(y))
    # print(Tw)
    if isPlotting:
        plt.figure()
        plt.plot(x, fnew(x), linewidth=3, alpha=1, linestyle='None', marker= '*', c='k', label ='Interpolate')
        plt.plot(x, y, linewidth=5, alpha=0.5, linestyle='-', c='r', label='Original')
        plt.legend()
        plt.xlabel('Radius[mm]')
        plt.ylabel(r'$\sigma_{rz}$'+r'[$N/mm^2$]', fontsize=12)
        plt.title('F='+str(Freq[p-1])+' [KHz]')
        # plt.plot(x, real(y), marker='o')
    return Tw
def funStressZZ(p, isPlotting=False):
    x,y=stressZZ(p)
    Freq = np.arange(5, 1000, 5)
    # fun= interp1d(x,y, kind='cubic')
    fnew=Spline(x,real(y), k=1)#interp1d(x,abs(y),fill_value=(0, 0),kind='cubic' ,bounds_error=False)#
    Tw=max(np.real(y))#np.sum(fnew(x))/len(x)#integrate.simps(fnew(x), x)#max(np.real(y))#
    if isPlotting:
        plt.figure()
        plt.plot(x, fnew(x), linewidth=3, alpha=1, linestyle='None', marker= '*', c='k', label ='Interpolate')
        plt.plot(x, y, linewidth=5, alpha=0.5, linestyle='-', c='r', label='Original')
        plt.legend()
        plt.xlabel('Radius[mm]')
        plt.ylabel(r'$\sigma_{zz}$'+r'[$N/mm^2$]', fontsize=12)
        plt.title('F='+str(Freq[p-1])+' [KHz]')
        # plt.plot(x, real(y), marker='o')
    return Tw
def t_w(isPlotting=True):
    Freq = np.arange(5, 1000, 5)
    AwR,AwZ=[],[]#np.zeros_like(Freq)
    for i in np.arange(0,199,1):
        AwR.append(funStressRZ(i+1,isPlotting=False))
        AwZ.append(funStressZZ(i+1,isPlotting=False))
        # print(funStressRZ(i+1,isPlotting=False))
    AwR=np.array(AwR)
    AwZ=np.array(AwZ)
    if isPlotting:
        plt.figure()
        plt.plot(Freq, abs(AwR), linewidth=3, alpha=1, linestyle='None', marker= '*', c='k', label ='Tw_rz')
        plt.plot(Freq, abs(AwZ), linewidth=3, alpha=1, linestyle='None', marker= '*', c='r', label ='Tw_zz')
        plt.legend()
        # plt.plot(x, y, linewidth=5, alpha=0.5, linestyle='-', c='r', label='Original')
    np.save("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\AwR", np.array(AwR))
    np.save("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\AwZ", np.array(AwZ))
    return AwR,AwZ
if __name__=='__main__':
    Freq = np.arange(5, 1000, 5)*1e3
    Aw=t_w()
    # Model=HM.t_w()
    
    # K =Model._tipDisp._equations.K[:,2]
    # y=scipy.special.jv(1,5e-3*K*0.93)
    # fy=interp1d(Model._tipDisp._equations.Freq, y)
    # plt.figure()
    # plt.plot(Freq,abs(fy(Freq)*Aw)*5)
    # funStressRZ(100,isPlotting=True)
    plt.show()
