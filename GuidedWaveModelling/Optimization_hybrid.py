import numpy as np
import Hybridmodel as HM
import Figure_plot as graph
import matplotlib.pyplot as plt
import scipy.special 
from scipy.special  import jv
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from sko.GA import GA, GA_TSP


def Freq_stress(isPlotting=True):
    x1,x2,x3=[-57.60997584214649, -47.62628935106598, 0.7088107000782904]#(11.82292584043821, 4.064344077718797, 0.9240273141674396) #| fitness : 0.00023270012712943655
# [5, -1.092104509176798, 0.9218205932997062] | fitness : 0.08528415610979505
    f=900e3
    # Importing the wave stress
    Freq = np.arange(5, 1000, 5)*1e3 # Hz
    p= np.argmin(abs(Freq-f))
    Rz_waveNumber=pd.read_csv("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\data_stress_RZ_waveNumber.csv")
    stress_KRz=Rz_waveNumber['sigma_RZ[N/mm^2] '+'F='+str(int(Freq[p-1]*1e-3))+' [KHz]']
    K_rz=Rz_waveNumber['K[rad/mm]']*1e3
    K=np.linspace(10,3000,100)
    stress_KRz=interp1d(K_rz, stress_KRz)(K)
    
    #############-----
    Model=HM.t_w()
    Ks=Model._tipDisp._equations.K[:,2]
    Ka=Model._tipDisp._equations.K[:,1]
    x=Model._tipDisp._equations.Freq
    a=Model._tipDisp._equations.a
    # Admittance term
    Aw=Model.constan_term(isPlotting=True)
    fAw=interp1d(x,Aw)
    Model.plotting_measured_Admittance()
    gamma=0.93
    t_r=x1*a*jv(1,K*a*x3)+x2*fAw(f)*a*jv(2,K*a*x3)/K
    print(np.sum((stress_KRz-t_r)**2))

    if isPlotting:
        fig,axes = plt.subplots(1,1, sharex=True)
        # graph.figureplot(Model1._tipDisp._equations.Freq,np.imag(t_w().Free_Admittance(t_w()._tipDisp._equations.Freq))
        graph.figureplot(K, abs(stress_KRz),ax=axes, label = str(Freq[p-1])+'[Hz]-FEM', ylabel='[Pa-m^2]')
        graph.figureplot(K, abs(t_r),ax=axes, label = str(Freq[p-1])+'[Hz]-Ana', ylabel='[Pa-m^2]')

def demo_func(p):
    x1,x2,x3=p
    f=10e3
    # Importing the wave stress
    Freq = np.arange(5, 1000, 5)*1e3 # Hz
    p= np.argmin(abs(Freq-f))
    Rz_waveNumber=pd.read_csv("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\data_stress_RZ_waveNumber.csv")
    stress_KRz=Rz_waveNumber['sigma_RZ[N/mm^2] '+'F='+str(int(Freq[p-1]*1e-3))+' [KHz]']
    K_rz=Rz_waveNumber['K[rad/mm]']*1e3
    K=np.linspace(10,3000,1000)
    stress_KRz=interp1d(K_rz, stress_KRz)(K)
    
    #############-----
    Model=HM.t_w()
    Ks=Model._tipDisp._equations.K[:,2]
    Ka=Model._tipDisp._equations.K[:,1]
    x=Model._tipDisp._equations.Freq
    a=Model._tipDisp._equations.a
    # Admittance term
    Aw=Model.constan_term(isPlotting=False)
    fAw=interp1d(x,Aw)
    t_r=x1*a*jv(1,K*a*x3)+x2*fAw(f)*a*jv(2,K*a*x3)/K

    return np.sum((stress_KRz-t_r)**2)

# driver code
if __name__=='__main__':
    Freq_stress()
    # Model1.constan_term(isPlotting=True)
    plt.show()
    # demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + (x[2] - 0.5) ** 2
    # ga = GA(func=demo_func, n_dim=3, size_pop=100, max_iter=5, lb=[-10, -10, 0], ub=[10, 10, 1],
        # precision=[1e-7, 1e-7, 1])
    # best_x, best_y = ga.run(10)
    # Y_history = pd.DataFrame(ga.all_history_Y)
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    # Y_history.min(axis=1).cummin().plot(kind='line')
    # plt.show()  
    # print('best_x:', best_x, '\n', 'best_y:', best_y)
 




