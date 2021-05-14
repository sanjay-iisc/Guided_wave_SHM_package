import sys
sys.path.append("./")
from GuidedWaveModelling import newGuidedwavePropagation as Analytical
import numpy as np
import matplotlib.pyplot as plt
Wave =Analytical.WaveDisplacment()
def stressfunction(k,a, isPlotting=True):
    sigma_r , sigma_z = np.zeros_like(k, dtype=complex),np.zeros_like(k, dtype=complex)
    for i, K in enumerate(k):
        stressMatrix=Wave.prhwStress_function( K , a)
        sigma_r[i]= stressMatrix[0]
        sigma_z[i]= stressMatrix[1]
    return sigma_r, sigma_z
def plotFigure(x,y,ax=None, Label='Stress'):
    ax.plot(x,y, label=Label)
    ax.legend()


if __name__=='__main__':
    K15mm = Wave.K
    F15mm= Wave.Freq
    i=2
    sigma15mmR,sigma15mmZ =stressfunction(K15mm[:,i],5)
    #----->>>>>>>>
    #------ Wave Number for the 125 micro mm
    K125mm = np.load('E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix125micro_mm.npy')
    F125mm= np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix125micro_mm.npy")
    sigma125mmR,sigma125mmZ =stressfunction(K125mm[:,i],5)
    #----->>>>>Plotting function--->>>

    plt.figure()
    axes=plt.gca()
    plotFigure(F15mm,abs(sigma15mmR),ax=axes, Label ='waveNumber for 1.5mm-A0' )
    plotFigure(F125mm,abs(sigma125mmR),ax=axes, Label ='waveNumber for 125mm-A0' )
    
    #----Plotting wavenumber
    plt.figure()
    axes=plt.gca()
    plotFigure(F15mm,abs(K15mm[:,i]),ax=axes, Label ='waveNumber for 1.5mm-A0' )
    plotFigure(F125mm,(K125mm[:,i]),ax=axes, Label ='waveNumber for 125mm-A0' )
    plt.show()