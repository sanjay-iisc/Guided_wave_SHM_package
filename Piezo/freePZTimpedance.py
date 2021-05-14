import sys
sys.path.append("./")
import numpy as np
import math
import cmath
import scipy.special as fun
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from numpy import imag,real
from GuidedWaveModelling import newGuidedwavePropagation as Analytical
# plt.style.use(['science'])
# plt.rcParams.update({'figure.figsize': (6.0, 4.0),'figure.dpi':500})

def Linear_Fit(X, Y):
    X = X.reshape(-1, 1)  ### making 2d array here
    model = LinearRegression().fit(X, Y)
    r_sq = model.score(X, Y)
    # print('coefficient of determination:', r_sq)
    # print('intercept:', new_model.intercept_)
    # print('slope:', new_model.coef_)
    y_pred = model.predict(X)
    # print("predicted response:", y_pred, sep="\n")

    return y_pred, model.coef_


def figureplot(x,y,ax=None,**keyword):
    """"""
    
    '''
    figNumber = is tupple(2,1)
    '''
    A=[]
    XLABEL='[MHz]'
    YLABEL='[A.U]'
    for key, value in keyword.items():
        # print('{}={}'.format(key , value))
        
        if key=='xlabel':
            XLABEL=keyword[key]
            A.append(key)
        elif key=='ylabel':
            YLABEL=keyword[key]
            A.append(key)
        elif key=='title':
            Title=keyword[key]
            A.append(key)
        elif key=='filename':
            FileName=keyword[key]
            A.append(key)
  
    for a in A:
        keyword.pop(a)
    
        
    ax.plot(x,y,**keyword )
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.legend()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    for a in A :
        if a == 'title':
            ax.set_title(Title)
        # elif a =='filename':
            # plt.savefig('K:\LMC\Sanjay\ComsolData\\NicolasResults\ppt_12042021\\Admittance\\'+FileName+'.png')

def freePiezoImpedance(Freq,isPlotting=False):
    # plt.figure()
    """
    args :
        Freq =in hz
    """
    
    #####Bonded
    dis = pd.read_csv("K:\LMC\Sanjay\Comsolresults\\NicolasResults\\NicolasResults4.csv", skiprows=4)

    bondedfreq=dis['freq (kHz)'].to_numpy()*1e3 #in Hz
    V=1
        
      
    current=(dis['Current (A), Boundary Probe 1']).str.replace('i','j').apply(lambda x: np.complex(x))
    bondedAd =np.array((current))
    #Free Impedance
    FreeImFile = pd.read_csv("K:\LMC\Sanjay\Comsolresults\\NicolasResults\\NicolasResults3_freeImpedance.csv", skiprows=4)
    freeFreq=FreeImFile['% freq (kHz)']*1e3
    freeIm=(FreeImFile['i*( es.nD*es.omega) (A), Boundary Probe 1']).str.replace('i','j').apply(lambda x: np.complex(x))
    data={'current':[],'Admittance':[],'Impedance':[],'ur':[], 'ur1':[], 'ur2':[]}
    UtipFEm=dis['Displacement field, R component (mm), Point: (5, 1.5)'].str.replace('i','j').apply(lambda x: np.complex(x))
    # UtipFEm=dis['Total displacement (mm), Point: (5.01, 1.5)']
    Freq=Freq*1e6#np.linspace(20e3,1000e3, 1000)#
    for freq in Freq :
        ### Frequency
        f=freq #hz ## frequency
        w = 2* np.pi * f
        
        zeta=0
        ### all are in MKS
        a = 5e-3#(6.98e-3)/2#5e-3#6.98e-3/2  ##m radius of piezo
        ta =0.125e-3#0.4e-3 #0.125e-3#0.216e-3 ## thickness of the pzt
        Volt = 1#10 ## voltage applied
        d31 =-171e-12# 175e-12#-190e-12#-175e-12 ##meter/v dielectric constent
        e0 =8.85*1e-12 # (N  / Vot^2) or Farades
        eps33=1790*e0*(1-1j*zeta)#(N / Vot^2)#1900*e0#1800*e0# ## epsilon proper normlised
        s11e =16.4e-12*(1-1j*zeta)#(M^2/N)#18e-12#1/(48*1e9)#18e-12 # 1/E11 for the 1d 
        rho=7750#(#Kg/M^3)7800 #7600 ## density of PZT
        s12e= 1/(48*1e9) #1
        nu_p=0.35#0.5#-s12e/s11e
        ### 
        A = np.pi* a**2 ## area of PZT
        E3 = -Volt/ta ## Electric feild in the PZT
        
        
        ## equation constent 
        c =np.sqrt(1/ ( s11e*rho *(1-nu_p**2) )) #np.sqrt(1/ ( s11e*rho )) ## wave velocity
        SISA = d31 * E3### Strain induced at the tip of the PZT
        UISA = SISA * a ### Displacment at the tip of the PZT
        k = w/c ## omega / c here w is omega not frequency
        K31 =d31**2 / (s11e* eps33) ## coupling cofficent of the PZT
        KP = np.sqrt( (2 * K31) / (1-nu_p))
        C = ( (eps33) *A)/ta ## capicitance of PZT# page 66 
        # print(C)
        phi = k*a
        ### Equation 
        #current
        ## edge displacment 
        ur_a1 = (1+nu_p) * fun.jv(1, phi)
        ur_a2 =  phi * fun.jv(0, phi) -(1-nu_p)*fun.jv(1,phi)
        ur_a = ur_a1/ ur_a2
        data['ur'].append(ur_a)
        data['ur1'].append(ur_a1)
        data['ur2'].append(ur_a2)
        
        ## current
        
        Icurrent = 1j * w *C  *(  1 - KP**2 * (1 - ur_a ))*Volt
        
        data['current'].append(Icurrent)
        ## Admitance
        
        Y = Icurrent/ Volt
        data['Admittance'].append(Y)
        ### Impedance 
        
        Z =1/ Y
        data['Impedance'].append(Z)
    
  

    #------- > slope of the  bonded pzt
    w_Bondede=bondedfreq*2*np.pi
    Omega= 2*np.pi*Freq

    yfdesh0 =  1j*C 
    ydesh0 =yfdesh0*(1-KP**2)
    #_------> derivate of the bonded pzt
    # 4. Spline derivative with smoothing set to 0.01
    # ydesh = dxdt(np.imag(bondedAd),w_Bondede,  kind="finite_difference", k=1)
    # yfdes=dxdt(imag(data['Admittance']),Omega,  kind="finite_difference", k=1)
    # ydesh =
    # #__--  test the interpolation
    # plt.figure()
    # plt.plot(w_Bondede,  ydesh)
    # plt.plot(Omega , yfdes)
    # #------- The S ratio for the calculation
    # Kp_real =real(KP)
    # C_real =real(C)
    # Yfdes_0 = C_real
    # S1 = (bondedAd)- w_Bondede *Yfdes_0*(1-Kp_real **2)
    # S2 = w_Bondede*C_real*Kp_real **2
    # S = S1/S2
    ####### utip from the
    S1=(bondedAd)- w_Bondede *(ydesh0)
    S2 = w_Bondede*(yfdesh0-ydesh0)
    S = S1/S2
    # ####
    # FemUtip_1= bondedAd- 1j*w_Bondede *Yfdes_0*(1-Kp_real **2)
    # FemUtip_2= 1j*w_Bondede*C_real*Kp_real **2
    # FemUtip= (FemUtip_1/FemUtip_2)*d31*a 

    AdmitanceFrom_tip = -1j*C *w_Bondede * (1- KP**2 * (1- ( (UtipFEm *1e-3*ta)/ (d31 * a *V ))))
    if isPlotting:
        #-----> Impedance curve-plot-1
        #### Real-Impedance curve
        fig,axes = plt.subplots(2,1, sharex=True)
        figureplot(Freq,np.real(data['Admittance']), ax=axes[0], xlabel='F[Hz]', ylabel='Real[Ad][Siemens]', label='Unbonded-Ana')
        figureplot(bondedfreq,np.real(bondedAd),ax=axes[0], label='Bonded-FEM')
        figureplot(freeFreq,np.real(freeIm),ax=axes[0], label='Unbonded-FEM',xlabel='F[Hz]', ylabel='Real[Ad][Siemens]')
        
        #### Imag-Impedance curve
        figureplot(Freq,np.imag(data['Admittance']), ax=axes[1], xlabel='F[Hz]', ylabel='Im[Ad][Siemens]', label='Unbonded-Ana')
        figureplot(bondedfreq,np.imag(bondedAd),ax=axes[1], label='Bonded-FEM')
        figureplot(freeFreq,np.imag(freeIm),ax=axes[1], label='Unbonded-FEM',xlabel='F[Hz]', ylabel='Imag[Ad][Siemens]',filename='AnaLytical-Impedance')
        
        # # plotting ------ >>>> S 
        fig,axes = plt.subplots(2,1, sharex=True)
        figureplot(bondedfreq,np.real(AdmitanceFrom_tip ),ax=axes[0], label='fromTip-FEM', linestyle='--')
        figureplot(bondedfreq,np.real(bondedAd),ax=axes[0], label='Bonded-FEM')
        figureplot(bondedfreq,np.imag(AdmitanceFrom_tip ),ax=axes[1], label='fromTip-FEM', linestyle='--')
        figureplot(bondedfreq,-np.imag(bondedAd),ax=axes[1], label='Bonded-FEM')
        # figureplot(bondedfreq,abs(real(UtipFEm)*1e-3),ax=axes, label ='Utip-FEM')

        # #-------------->>> Ploting the tip dispalcment form the the Admittance 
        Wave =Analytical.WaveDisplacment()
        AnalyticalTip=Wave.PRHW()
        fig,axes = plt.subplots(1,1, sharex=True)
        # figureplot(Wave.Freq*1e6,abs(AnalyticalTip),ax=axes, label='Ana-tip')
        # figureplot(bondedfreq,np.real(AdmitanceFrom_tip ),ax=axes, label='fromTip-FEM', linestyle='--')
        figureplot(bondedfreq,abs((UtipFEm*1e3*ta)/ (d31*Volt*a)),ax=axes, label ='Utip-FEM')
        #----> plotting the boned pzt
        fig,axes = plt.subplots(1,1, sharex=True)
        figureplot(bondedfreq,np.abs(S),ax=axes, label='Bonded')
    return bondedfreq,S#abs((UtipFEm)*1e-3)#abs(FemUtip)#(S*d31)#abs(UtipFEm)*1e-3 #-(S*d31*a / ta)
if __name__=='__main__':
    freq=np.linspace(0.005,1, 200)
    F,S=freePiezoImpedance(freq,isPlotting=True)
    plt.show()
    