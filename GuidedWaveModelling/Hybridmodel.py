
# References for equations
# [1] : Tuned Lamb wave excitation and detection with piezoelectric wafer active sensors for structural health monitoring 
# [2] : Lamb wave tuning curve calibration for surface-bonded piezoelectric transducers
# [3] : Hybrid empirical/analytical modeling of guided wave generation by circular piezoceramics 
# [4] : Book (giurgiutiu2007structural) page no 483 Giurgiutiu, V.Structural health monitoring: with piezoelectric wafer active sensors Elsevier, 2007

from __future__ import division
import sys
sys.path.append("./")
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from math import log, ceil, inf
import scipy
from scipy.special import jv,jn_zeros
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy import interpolate
import pandas  as pd 
import os 
from Piezo import freePZTimpedance as freePZT
from sklearn.linear_model import LinearRegression
import GuidedWaveModelling.Figure_plot as graph
### 
class WaveField:
    my_dict_input={"plate_properties":{ "d":(1.5/2)*1e-3,"E":70e9,"nu":0.33,"rho":2700},
"pzt_properties": {"a":5e-3,"Volt":1,"hp":0.125e-3,"d31":-175e-12,"eps33":1790*8.85*1e-12*(1-1j*0.05)
,"rhoPiezo":7750,"nu_p":0.35,"s11e":16.4e-12}}#*(1-1j*0.05)

    my_dict_waveNumber={'K':np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix.npy")*1e3,
    'Freq':np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix.npy")*1e6}

    my_dict_stress_parameter={'alpha_r': 1.11,'beta_r':-0.22,'shearLag_r':0.93,
    'alpha_z': 1.17,'beta_z':0.41,'shearLag_z':0.92,'zeta':0.14}

    def __init__(self):
        for name in WaveField.my_dict_input:
            for key in WaveField.my_dict_input[name]:
                setattr(self, key, WaveField.my_dict_input[name][key])
        
        for key in WaveField.my_dict_waveNumber:
            setattr(self, key, WaveField.my_dict_waveNumber[key])
        
        for key in WaveField.my_dict_stress_parameter:
            setattr(self, key, WaveField.my_dict_stress_parameter[key])
        
        self.omega=2*np.pi*self.Freq #rad/s
        self.obs_r=25e-3
        self.saveFigure='E:\PPT\Presentation\\02052021_ppt\Figure\\'
        self.Nmodes=4
        # Bulk wave speed
        self.Lambada = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        self.Mu = self.E / (2 * (1 + self.nu))
        self.C_L = math.sqrt((self.Lambada + 2 * self.Mu) / self.rho)
        self.C_T = math.sqrt(self.Mu / self.rho)
        
    # for plotting the imported wavenumber 
    def plottingWaveNumber(self): 
        fig,axes = plt.subplots(1,1, sharex=True)
        for i in range(self.Nmodes):
            graph.figureplot(self.Freq,np.real(self.K[:,i]), ax=axes, xlabel='F[Hz]', ylabel='K[rad/m]', title ='Wave Number-1.5mmplate',linestyle='None', marker='o', markersize=2,label='Mode :'+str(i))
            graph.figureplot(self.Freq,np.imag(self.K[:,i]), ax=axes, xlabel='F[Hz]', ylabel='K[rad/m]', title ='Wave Number-1.5mmplate',linestyle='None', marker='o', markersize=2,c='r',label='Mode :'+str(i))
    
    def da_dash(self, K , W ):
        # Derivative of equ 20 of [3]
        p = np.lib.scimath.sqrt((W / self.C_L) ** 2 - K ** 2)
        q = np.lib.scimath.sqrt((W / self.C_T) ** 2 - K ** 2)
        ##Equation Da1 and Da2
        # Da1= np.cos(p*self.d)*np.sin(q*self.d) * 4* K**2 * p * q
        Da1_1= np.cos(p*self.d)*np.sin(q*self.d) * 8* K * p * q
        Da1_2= np.cos(p*self.d)*np.sin(q*self.d) * 4* K**2 * (-K/p) * q      
        Da1_3= np.cos(p*self.d)*np.sin(q*self.d) * 4* K**2 * (-K/q) * p     
        Da1_4= (K*self.d/p)*np.sin(p*self.d)*np.sin(q*self.d) * 4* K**2 * p * q 
        Da1_5= -(K*self.d/q)*np.cos(p*self.d)*np.cos(q*self.d) * 4* K**2 * p * q 
        #Da2=np.cos(q*self.d)*np.sin(p*self.d) *(q**2-k**2)**2
        Da2_1=np.cos(q*self.d)*np.sin(p*self.d) *(-8*K)*(q**2-K**2)
        Da2_2=(K*self.d/q)*np.sin(q*self.d)*np.sin(p*self.d) *(q**2-K**2)**2
        Da2_3=-(K*self.d/p)*np.cos(q*self.d)*np.cos(p*self.d) *(q**2-K**2)**2
        Da=Da1_1+Da1_2+Da1_3+Da1_4+Da1_5+Da2_1+Da2_2+Da2_3
        
        return Da
    
    def ds_dash(self, K , W ):
        # Derivative of equ 19 of [3]
        p = np.lib.scimath.sqrt((W / self.C_L) ** 2 - K ** 2)
        q = np.lib.scimath.sqrt((W / self.C_T) ** 2 - K ** 2)
        ##Equation Da1 and Da2
        # Ds1= np.sin(p*self.d)*np.cos(q*self.d) * 4* K**2 * p * q
        Ds1_1= np.sin(p*self.d)*np.cos(q*self.d) * 8* K * p * q
        Ds1_2= np.sin(p*self.d)*np.cos(q*self.d) * 4* K**2 * (-K/p) * q      
        Ds1_3= np.sin(p*self.d)*np.cos(q*self.d) * 4* K**2 * (-K/q) * p     
        Ds1_4= -(K*self.d/p)*np.cos(p*self.d)*np.cos(q*self.d) * 4* K**2 * p * q 
        Ds1_5= (K*self.d/q)*np.sin(p*self.d)*np.sin(q*self.d) * 4* K**2 * p * q 
        #Ds2=np.sin(q*self.d)*np.cos(p*self.d) *(q**2-k**2)**2
        Ds2_1=np.sin(q*self.d)*np.cos(p*self.d) *(-8*K)*(q**2-K**2)
        Ds2_2=-(K*self.d/q)*np.cos(q*self.d)*np.cos(p*self.d) *(q**2-K**2)**2
        Ds2_3=(K*self.d/p)*np.sin(q*self.d)*np.sin(p*self.d) *(q**2-K**2)**2
        Ds=Ds1_1+Ds1_2+Ds1_3+Ds1_4+Ds1_5+Ds2_1+Ds2_2+Ds2_3
        
        return Ds

    def da_equation(self,K,W):
        # eq 20 [3]
        C_L = self.C_L# mm/microsec
        C_T = self.C_T # mm/microsec
        W=W #Mhz
        D= self.d
        K= K# rad/mm
        p = np.lib.scimath.sqrt((W / C_L) ** 2 - K ** 2)
        q = np.lib.scimath.sqrt((W / C_T) ** 2 - K ** 2)
        Da1 = np.cos(p * D ) * np.sin(q * D ) * 4 * (K ** 2) * p * q

        Da2 = np.cos(q * D ) * np.sin(p * D) * ( q ** 2 - K ** 2 ) ** 2
        return Da1+Da2
    
    def ds_equation(self,K,W):
        # eq 19 [3]
        C_L = self.C_L# mm/microsec
        C_T = self.C_T # mm/microsec
        W=W #Mhz
        D= self.d
        K= K# rad/mm
        p = np.lib.scimath.sqrt((W / C_L) ** 2 - K ** 2)
        q = np.lib.scimath.sqrt((W / C_T) ** 2 - K ** 2)
        Ds1= np.sin(p * D) * np.cos(q * D) * 4 * (K ** 2) * p * q
    
        Ds2=np.sin(q * D) * np.cos(p * D) * ( q ** 2-K ** 2 ) ** 2
        return Ds2+Ds1

    def N_sym(self, K , W ):
        '''
        Args : K = waveNumber
        a = matrix 
        '''
        p = np.lib.scimath.sqrt((W / self.C_L) ** 2 - K ** 2)
        q = np.lib.scimath.sqrt((W / self.C_T) ** 2 - K ** 2)
        
        l1= np.sin( p* self.d )*np.sin( q* self.d )
        l2= np.cos( p* self.d )*np.cos( q* self.d )
        l3 = np.sin( p* self.d )*np.cos( q* self.d)
        l4 = np.cos( p* self.d )*np.sin( q* self.d )
        
        ## 
        a1 =  ( q**2 + K**2 )
        a2 =  ( K**2 - q**2 )
        
        ##
        N11 = -q* a1 *l2 
        N12 = -K * a2 *l4 - 2 * p * q * K * l3
        
        N21 = -K * a2 *l4 - 2 * p * q * K * l3 
        N22 = p* a1 * l1
        
        ## 
        
        Ns = np.array( [[N11 , N12 ], [N21 , N22 ]], dtype =complex)
        
        return Ns
    
    def N_Antisym(self, K , W, angle = np.deg2rad(0)):
        '''
        Args : K = waveNumber
        a = matrix 
        '''
        p = np.lib.scimath.sqrt((W / self.C_L) ** 2 - K ** 2)

        q = np.lib.scimath.sqrt((W / self.C_T) ** 2 - K ** 2)
                
        l1= np.sin( p* self.d )*np.sin( q* self.d )
        l2= np.cos( p* self.d )*np.cos( q* self.d )
        l3 = np.sin( p* self.d )*np.cos( q* self.d)
        l4 = np.cos( p* self.d)*np.sin( q* self.d)
        
        ## 
        a1 =  ( q**2 + K**2 )
        a2 =  ( K**2 - q**2 )
        
        ##
        N11 = q * a1 *l1 
        N12 = -K * a2 *l3 - 2 * p * q * K * l4
        
        N21 = -K * a2 *l3 - 2 * p * q * K * l4 
        N22 = -p* (q**2 + K**2) * l2
        
        Na = np.array( [[N11 , N12 ], [N21 , N22 ]], dtype =complex)
    
        return Na

    def HankelMatrix(self , K, r):
        # Equation 23 of [3]
        '''
        K = Wave number 
        r = distance in mm from the excitation
        '''
        H11= scipy.special.hankel2(1,K * r) 
        H12 = 0
        H21 = 0
        H22= scipy.special.hankel2(1, K * r) 
        H_matrix  = np.array(  [ [H11 , H12]  , [H21 , H22]  ] , dtype = complex )
        return H_matrix
    
    def HankelMatrix_waveField (self , K, r):
        ## HankelMatrix has both h21 but here we have to have h21 and H20 as described in the
        # Equation 23 of [3]
        '''
        K = Wave number 
        r = distance in mm from the excitation
        '''
        H11= scipy.special.hankel2(1,K * r) 
        H12 = 0
        H21 = 0
        H22= scipy.special.hankel2(0, K * r) 
        H_matrix  = np.array(  [ [H11 , H12]  , [H21 , H22]  ] , dtype = complex )
        return H_matrix

    def Stress_function(self, K, a ):
        # right hand side of vector of equation 27 of [3]
        '''
        K = Wave number in rad / meter
        a = Radius of Piezo in m
    
        '''
        t11= scipy.special.jv(self.alpha_r, self.shearLag_r* K * a) * ((K/(1))  * a)**self.beta_r
        t22= self.zeta *  scipy.special.jv(self.alpha_z, self.shearLag_z*K *a) * ((K/(1))* a)**self.beta_z
        stressMatrix = np.array ([t11 , t22],dtype = complex)
        Mat_stress=stressMatrix.reshape((2,1))
        return Mat_stress
    def S0_Stress_from_FEM(self,f):
        FemFreq = np.arange(5, 1000, 5)*1e3
        t11=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSR.npy"))
        t22=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSZ.npy"))
        stressMatrix = np.array ([t11(f) , t22(f)],dtype = complex)
        Mat_stress=stressMatrix.reshape((2,1))
        return Mat_stress
    def A0_Stress_from_FEM(self,f):
        FemFreq = np.arange(5, 1000, 5)*1e3
        t11=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAR.npy"))
        t22=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAZ.npy"))
        stressMatrix = np.array ([t11(f) , t22(f)],dtype = complex)
        Mat_stress=stressMatrix.reshape((2,1))
        return Mat_stress   

    def S0_Stress_from_optimized(self,f):
        FemFreq =np.arange(5, 1000, 20)*1e3
        t11=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSR_optimized.npy"))
        t22=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSZ_optimized.npy"))
        stressMatrix = np.array ([t11(f) , t22(f)],dtype = complex)
        Mat_stress=stressMatrix.reshape((2,1))
        return Mat_stress
    def A0_Stress_from_optimized(self,f):
        FemFreq = np.arange(5, 1000, 20)*1e3
        t11=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAR_optimized.npy"))
        t22=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAZ_optimized.npy"))
        stressMatrix = np.array ([t11(f) , t22(f)],dtype = complex)
        Mat_stress=stressMatrix.reshape((2,1))
        return Mat_stress   
        
    

class tip_Displacement:
    def __init__(self):
        self._equations=WaveField()
    ## Symmetric Displacement
    def symDisplacement(self,k, omega):
        """ if is stress is 1 then stress multiply by hp else 0 it is without hp multiply"""
        Dis_S1=np.zeros_like(omega, dtype=complex)
        Dis_S2=np.zeros_like(omega, dtype=complex)
        #ur and uz
        for i, ks in enumerate(k):
            dss=self._equations.ds_dash(ks, omega[i])#scalar ; denominator of eq 23 of [3]
            Ns=self._equations.N_sym( ks ,omega[i]) # matrix 2*2; numerator of eq 23 of [3]
            Amp_SS = (Ns)*(ks/dss) # Matrix 2*2 ; fraction of equ 23
            Hs=self._equations.HankelMatrix(ks,self._equations.a) # Matrix 2*2 # Eq 23
            Const_S = np.matmul(Hs,Amp_SS )
            T_S=self._equations.Stress_function(ks,self._equations.a )# to check
            Dis_S1[i]=Const_S.dot((T_S))[0]
            Dis_S2[i]=Const_S.dot((T_S))[1]
        return Dis_S1,Dis_S2
    ## Antisymmetric Displacement 
    def antisymmetricDisplacement(self,k, omega):
        """ if is stress is 0 then stress multiply by hp else 1 it is without hp multiply"""
        Dis_A1=np.zeros_like(omega, dtype=complex)
        Dis_A2=np.zeros_like(omega, dtype=complex)
        for i, ka in enumerate(k):
            daa=self._equations.da_dash(ka, omega[i])#one value
            Na=self._equations.N_Antisym( ka ,omega[i]) # matrix 2*2
            Amp_AA = (Na)*(ka/daa) # Matrix 2*2
            Ha=self._equations.HankelMatrix(ka,self._equations.a)
            Const_A = np.matmul(Ha,Amp_AA )
            T_A=self._equations.Stress_function(ka,self._equations.a )
            Dis_A1[i]=Const_A.dot((T_A))[0]
            Dis_A2[i]=Const_A.dot((T_A))[1]
        return Dis_A1,Dis_A2
    
    def summation_two_modes(self,indexS=[2],indexA=[3]):
        # n is number of modes
        # unit -(rad/m) * (1/(N/m^2))
        # S mode
        temp_r, temp_z=0,0
        for ns in indexS:
            ks=self._equations.K[:,ns]
            w =self._equations.omega
            Urs,Uzs=self.symDisplacement(ks,w)
            temp_r+=Urs
            temp_z+=Uzs
        for na in indexA:
            ka=self._equations.K[:,na]
            w =self._equations.omega
            Ura,Uza=self.antisymmetricDisplacement(ka,w)
            temp_r+=Ura
            temp_z+=Uza
        return (temp_r*np.pi*1j)/(2*self._equations.Mu) # rad -m

class t_w:
    #### Importing the Admittance Curve from the FEM
    Comsol_Path="K:\LMC\Sanjay\Comsolresults\\NicolasResults\\NicolasResults4.csv"
        # Import the Free and Bonded Impedance for computing 
    Data = pd.read_csv(Comsol_Path, skiprows=4)
    FreqFEM=Data['freq (kHz)'].to_numpy()*1e3 #in Hz
    current=(Data ['Current (A), Boundary Probe 1']).str.replace('i','j').apply(lambda x: np.complex128(x)) # at 1 volt
    bonded_AD=current # I/V for Admittance
    Bonded_Admittance=interp1d(FreqFEM,bonded_AD)
    ### Importing for  Unbonded 
    FreeImFile = pd.read_csv("K:\LMC\Sanjay\Comsolresults\\NicolasResults\\NicolasResults3_freeImpedance.csv", skiprows=4)
    freeFreq=FreeImFile['% freq (kHz)']*1e3
    freeIm=(FreeImFile['i*( es.nD*es.omega) (A), Boundary Probe 1']).str.replace('i','j').apply(lambda x: np.complex128(x))
    Free_Admittance= interp1d(freeFreq,freeIm)

    def __init__(self):
        self._tipDisp=tip_Displacement()
    
    def constan_term(self, isPlotting=False):
        ### Calculation of Slope value
        fff= np.linspace(10e3, 100e3,1000)
        ### Free Admittance Slope
        yf_pred,yf_des=self.Linear_Fit(fff, np.imag(t_w().Free_Admittance(fff) )
        ,self._tipDisp._equations.Freq)
        ### Bonded Admittance Slope
        y_pred,y_des=self.Linear_Fit(fff, np.imag(t_w().Bonded_Admittance(fff) )
        ,self._tipDisp._equations.Freq)
        # print('Slope of Y {}'.format(y_des))
        # print('Slope of Yf {}'.format(yf_des))
        ### Equation Described in #[3] at 29
        ## 1j is multiplying because the of yf_des and Y_des is become real but we need imaginar
        YY1= np.imag(t_w().Bonded_Admittance(self._tipDisp._equations.Freq))-self._tipDisp._equations.omega*y_des
        YY2= self._tipDisp._equations.omega*yf_des-self._tipDisp._equations.omega*y_des
        # YY1= np.imag(t_w().Bonded_Admittance(self._tipDisp._equations.Freq))-self._tipDisp._equations.omega*1j*yf_des
        # +self._tipDisp._equations.omega*1j*yf_des*0.35**2
        # YY2=self._tipDisp._equations.omega*1j*yf_des*0.35**2
        YY=(YY1/YY2)#*self._tipDisp._equations.a*self._tipDisp._equations.d31
        #### Tip displacement
        PRHW=self._tipDisp.summation_two_modes()*self._tipDisp._equations.hp
        if isPlotting:
            fig,axes = plt.subplots(1,1, sharex=True)
            graph.figureplot(t_w()._tipDisp._equations.Freq,abs(YY),ax=axes, label='YY')
            # fig,axes = plt.subplots(1,1, sharex=True)
            # graph.figureplot(t_w()._tipDisp._equations.Freq,abs(PRHW),ax=axes, title='PRHW')
            fig,axes = plt.subplots(1,1, sharex=True)
            graph.figureplot(t_w()._tipDisp._equations.Freq,abs(abs(YY)/abs(PRHW)),ax=axes, title='YY/PRHW')
            # graph.figureplot(t_w()._tipDisp._equations.Freq,abs(PRHW),ax=axes, title='YY/PRHW')
        return abs(YY)
    def Linear_Fit(self,X, Y,Xnew):
        X = X.reshape(-1, 1)  ### making 2d array here
        model = LinearRegression().fit(X, Y)
        r_sq = model.score(X, Y)
        Xnew = Xnew.reshape(-1, 1)
        # print('coefficient of determination:', r_sq)
        # print('intercept:', new_model.intercept_)
        # print('slope:', new_model.coef_)
        y_pred = model.predict(Xnew)
        # print("predicted response:", y_pred, sep="\n")
        return y_pred, model.coef_[0]
    
    @staticmethod
    def plotting_measured_Admittance():
        fig,axes = plt.subplots(2,1, sharex=True)
        graph.figureplot(t_w()._tipDisp._equations.Freq,np.real(t_w().Bonded_Admittance(t_w()._tipDisp._equations.Freq))
        , ax=axes[0],
         xlabel='F[Hz]', ylabel='$|Y|[\Omega^{-1}]$', title ='Admittance curve for hp=125'+r'$\mu mm$',
         linestyle='None', marker='o', markersize=2, label ='Real-Bonded')
        graph.figureplot(t_w()._tipDisp._equations.Freq,np.imag(t_w().Bonded_Admittance(t_w()._tipDisp._equations.Freq)), 
        ax=axes[1],
         xlabel='F[Hz]', ylabel='$|Y|[\Omega^{-1}]$', title ='Admittance curve for hp=125'+r'$\mu mm$',
         linestyle='None', marker='o', markersize=2, label ='Imag-Bonded')

         ### Unbonded
        graph.figureplot(t_w()._tipDisp._equations.Freq,np.real(t_w().Free_Admittance(t_w()._tipDisp._equations.Freq)), 
        ax=axes[0],
         xlabel='F[Hz]', ylabel='$|Y|[\Omega^{-1}]$', title ='Admittance curve for hp=125'+r'$\mu mm$',
         linestyle='None', marker='o', markersize=2, label ='Real-UnBonded')
        graph.figureplot(t_w()._tipDisp._equations.Freq,np.imag(t_w().Free_Admittance(t_w()._tipDisp._equations.Freq)), 
        ax=axes[1],
         xlabel='F[Hz]', ylabel='$|Y|[\Omega^{-1}]$', title ='Admittance curve for hp=125'+r'$\mu mm$',
         linestyle='None', marker='o', markersize=2, label ='Imag-Unbonded')

class Displacement_Field_FEM:
    # Importing FEM results
     #### Importing the Admittance Curve from the FEM
    
    Comsol_Path="K:\LMC\Sanjay\Comsolresults\\NicolasResults\\NicolasResults4.csv"
    Data = pd.read_csv(Comsol_Path, skiprows=4)
    FemFreq =Data['freq (kHz)'].to_numpy()*1e3 #in Hz
    UrS0=Data['S0_u']*1e-3 #in m
    UrA0=Data['A0_u']*1e-3#in m
    UzS0=Data['S0_w']*1e-3 #in m
    UzA0=Data['A0_w']*1e-3#in m
    f_UrS0=interp1d(FemFreq,UrS0)
    f_UrA0=interp1d(FemFreq,UrA0)
    f_UzS0=interp1d(FemFreq,UzS0)
    f_UzA0=interp1d(FemFreq,UzA0)
    

    def __init__(self):
        self._equations=WaveField()
        #Importing
        FemFreq = np.arange(5, 1000, 5)*1e3
        FAwR=Spline(FemFreq,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\AwR.npy")*self._equations.a**2*1e6)
        FAwZ=Spline(FemFreq,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\AwZ.npy")*self._equations.a**2*1e6)
        # np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\AwZ")
        
        # self.tw_r=FAwR(self._equations.Freq)
        # self.tw_z=FAwZ(self._equations.Freq)
    ## Symmetric Displacement
    def symDisplacement(self,k, omega):
        """ if is stress is 1 then stress multiply by hp else 0 it is without hp multiply"""
        Dis_S1=np.zeros_like(omega, dtype=complex)
        Dis_S2=np.zeros_like(omega, dtype=complex)
        #ur and uz
        for i, ks in enumerate(k):
            dss=self._equations.ds_dash(ks, omega[i])#scalar ; denominator of eq 23 of [3]
            Ns=self._equations.N_sym( ks ,omega[i]) # matrix 2*2; numerator of eq 23 of [3]
            Amp_SS = (Ns)*(ks/dss) # Matrix 2*2 ; fraction of equ 23
            Hs=self._equations.HankelMatrix_waveField (ks,self._equations.obs_r) # Matrix 2*2 # Eq 23
            Const_S = np.matmul(Hs,Amp_SS )
            T_S=self._equations.S0_Stress_from_optimized(self._equations.Freq[i])#self._equations.Stress_function(ks,self._equations.a )# to check
            Dis_S1[i]=Const_S.dot((T_S))[0]#*self.tw_r[i]
            Dis_S2[i]=Const_S.dot((T_S))[1]#*self.tw_z[i]
        return Dis_S1,Dis_S2
    ## Antisymmetric Displacement 
    def antisymmetricDisplacement(self,k, omega):
        """ if is stress is 0 then stress multiply by hp else 1 it is without hp multiply"""
        Dis_A1=np.zeros_like(omega, dtype=complex)
        Dis_A2=np.zeros_like(omega, dtype=complex)
        for i, ka in enumerate(k):
            daa=self._equations.da_dash(ka, omega[i])#one value
            Na=self._equations.N_Antisym( ka ,omega[i]) # matrix 2*2
            Amp_AA = (Na)*(ka/daa) # Matrix 2*2
            Ha=self._equations.HankelMatrix_waveField (ka,self._equations.obs_r)
            Const_A = np.matmul(Ha,Amp_AA )
            T_A=self._equations.A0_Stress_from_optimized(self._equations.Freq[i])#self._equations.Stress_function(ka,self._equations.a )
            Dis_A1[i]=Const_A.dot((T_A))[0]#*self.tw_r[i]
            Dis_A2[i]=Const_A.dot((T_A))[1]#*self.tw_z[i]
        return Dis_A1,Dis_A2
    def Hybrid_Displacement(self,indexS=[2],indexA=[1], isPlotting=False):
        # n is number of modes
        # unit -(rad/m) * (1/(N/m^2))
        # S mode
        for ns in indexS:
            ks=self._equations.K[:,ns]
            w =self._equations.omega
            Urs,Uzs=self.symDisplacement(ks,w)
        for na in indexA:
            ka=self._equations.K[:,na]
            w =self._equations.omega
            Ura,Uza=self.antisymmetricDisplacement(ka,w)
        if isPlotting:
            
            fig,axes = plt.subplots(1,2, sharex=True,sharey=False)
            graph.figureplot(self._equations.Freq,abs((Urs*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[0], label='S0-Hybrid',c='k')
            graph.figureplot(self._equations.Freq,abs((Ura*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[0], label='A0-Hybrid',c='r')
            graph.figureplot(self._equations.Freq,Displacement_Field_FEM.f_UrS0(self._equations.Freq) , ax=axes[0], label='S0-FEM',
            linestyle='None',marker='*',c='k')
            graph.figureplot(self._equations.Freq,Displacement_Field_FEM.f_UrA0(self._equations.Freq) , ax=axes[0], label='A0-FEM',
            linestyle='None',marker='*',c='r')


            #--------------zzzz
            graph.figureplot(self._equations.Freq,abs((Uzs*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[1], label='S0-Hybrid',c='k')
            graph.figureplot(self._equations.Freq,abs((Uza*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[1], label='A0-Hybrid',c='r')
            graph.figureplot(self._equations.Freq,Displacement_Field_FEM.f_UzS0(self._equations.Freq) , ax=axes[1], label='S0-FEM',
            linestyle='None',marker='*',c='k')
            graph.figureplot(self._equations.Freq,Displacement_Field_FEM.f_UzA0(self._equations.Freq) , ax=axes[1], label='A0-FEM',
            linestyle='None',marker='*',c='r')
        #  xlabel='F[Hz]', ylabel='$|Y|[\Omega^{-1}]$', title ='Admittance curve for hp=125'+r'$\mu mm$',
        #  linestyle='None', marker='o', markersize=2, label ='Real-Bonded')
        return (Urs*np.pi*1j)/(2*self._equations.Mu),(Uzs*np.pi*1j)/(2*self._equations.Mu), 
        (Ura*np.pi*1j)/(2*self._equations.Mu),(Uza*np.pi*1j)/(2*self._equations.Mu)   # rad -m

#_____________________Average stress-----------------------------------------------
class Displacement_Field_Avarage:
    # Importing FEM results
     #### Importing the Admittance Curve from the FEM
    
    Comsol_Path="K:\LMC\Sanjay\Comsolresults\\NicolasResults\\NicolasResults4.csv"
    Data = pd.read_csv(Comsol_Path, skiprows=4)
    FemFreq =Data['freq (kHz)'].to_numpy()*1e3 #in Hz
    UrS0=Data['S0_u']*1e-3 #in m
    UrA0=Data['A0_u']*1e-3#in m
    UzS0=Data['S0_w']*1e-3 #in m
    UzA0=Data['A0_w']*1e-3#in m
    f_UrS0=interp1d(FemFreq,UrS0)
    f_UrA0=interp1d(FemFreq,UrA0)
    f_UzS0=interp1d(FemFreq,UzS0)
    f_UzA0=interp1d(FemFreq,UzA0)
    

    def __init__(self):
        self._equations=WaveField()
        #Importing
        FemFreq = np.arange(5, 1000, 5)*1e3
        FAwR=Spline(FemFreq,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\AwR.npy")*self._equations.a*1e3)
        FAwZ=Spline(FemFreq,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\AwZ.npy")*self._equations.a*1e3)
        # np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\AwZ")
        
        self.tw_r=FAwR(self._equations.Freq)
        self.tw_z=FAwZ(self._equations.Freq)
    ## Symmetric Displacement
    def symDisplacement(self,k, omega):
        """ if is stress is 1 then stress multiply by hp else 0 it is without hp multiply"""
        Dis_S1=np.zeros_like(omega, dtype=complex)
        Dis_S2=np.zeros_like(omega, dtype=complex)
        #ur and uz
        for i, ks in enumerate(k):
            dss=self._equations.ds_dash(ks, omega[i])#scalar ; denominator of eq 23 of [3]
            Ns=self._equations.N_sym( ks ,omega[i]) # matrix 2*2; numerator of eq 23 of [3]
            Amp_SS = (Ns)*(ks/dss) # Matrix 2*2 ; fraction of equ 23
            Hs=self._equations.HankelMatrix_waveField (ks,self._equations.obs_r) # Matrix 2*2 # Eq 23
            Const_S = np.matmul(Hs,Amp_SS )
            T_S=self._equations.Stress_function(ks,self._equations.a )# to check
            Dis_S1[i]=Const_S.dot((T_S))[0]*self.tw_r[i]
            Dis_S2[i]=Const_S.dot((T_S))[1]*self.tw_z[i]
        return Dis_S1,Dis_S2
    ## Antisymmetric Displacement 
    def antisymmetricDisplacement(self,k, omega):
        """ if is stress is 0 then stress multiply by hp else 1 it is without hp multiply"""
        Dis_A1=np.zeros_like(omega, dtype=complex)
        Dis_A2=np.zeros_like(omega, dtype=complex)
        for i, ka in enumerate(k):
            daa=self._equations.da_dash(ka, omega[i])#one value
            Na=self._equations.N_Antisym( ka ,omega[i]) # matrix 2*2
            Amp_AA = (Na)*(ka/daa) # Matrix 2*2
            Ha=self._equations.HankelMatrix_waveField (ka,self._equations.obs_r)
            Const_A = np.matmul(Ha,Amp_AA )
            T_A=self._equations.Stress_function(ka,self._equations.a )
            Dis_A1[i]=Const_A.dot((T_A))[0]*self.tw_r[i]
            Dis_A2[i]=Const_A.dot((T_A))[1]*self.tw_z[i]
        return Dis_A1,Dis_A2
    def Hybrid_Displacement(self,indexS=[2],indexA=[1], isPlotting=False):
        # n is number of modes
        # unit -(rad/m) * (1/(N/m^2))
        # S mode
        for ns in indexS:
            ks=self._equations.K[:,ns]
            w =self._equations.omega
            Urs,Uzs=self.symDisplacement(ks,w)
        for na in indexA:
            ka=self._equations.K[:,na]
            w =self._equations.omega
            Ura,Uza=self.antisymmetricDisplacement(ka,w)
        if isPlotting:
            
            fig,axes = plt.subplots(1,2, sharex=True,sharey=False)
            graph.figureplot(self._equations.Freq,abs((Urs*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[0], label='S0-Hybrid',c='k')
            graph.figureplot(self._equations.Freq,abs((Ura*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[0], label='A0-Hybrid',c='r')
            graph.figureplot(self._equations.Freq,Displacement_Field_Avarage.f_UrS0(self._equations.Freq) , ax=axes[0], label='S0-FEM',
            linestyle='None',marker='*',c='k')
            graph.figureplot(self._equations.Freq,Displacement_Field_Avarage.f_UrA0(self._equations.Freq) , ax=axes[0], label='A0-FEM',
            linestyle='None',marker='*',c='r')


            #--------------zzzz
            graph.figureplot(self._equations.Freq,abs((Uzs*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[1], label='S0-Hybrid',c='k')
            graph.figureplot(self._equations.Freq,abs((Uza*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[1], label='A0-Hybrid',c='r')
            graph.figureplot(self._equations.Freq,Displacement_Field_Avarage.f_UzS0(self._equations.Freq) , ax=axes[1], label='S0-FEM',
            linestyle='None',marker='*',c='k')
            graph.figureplot(self._equations.Freq,Displacement_Field_Avarage.f_UzA0(self._equations.Freq) , ax=axes[1], label='A0-FEM',
            linestyle='None',marker='*',c='r')
        #  xlabel='F[Hz]', ylabel='$|Y|[\Omega^{-1}]$', title ='Admittance curve for hp=125'+r'$\mu mm$',
        #  linestyle='None', marker='o', markersize=2, label ='Real-Bonded')
        return (Urs*np.pi*1j)/(2*self._equations.Mu),(Uzs*np.pi*1j)/(2*self._equations.Mu), 
        (Ura*np.pi*1j)/(2*self._equations.Mu),(Uza*np.pi*1j)/(2*self._equations.Mu)   # rad -m
    
# driver code
if __name__=='__main__':
    # Try=tip_Displacement()
    # Try._equations.plottingWaveNumber()
    # Try._equations.omega
    # Utip1=Try.summation_two_modes(indexS=[2],indexA=[1,3])
    # Utip2=Try.summation_two_modes(indexS=[2],indexA=[1,3])
    # fig,axes = plt.subplots(1,1, sharex=True)
    # graph.figureplot(Try._equations.Freq,abs(Utip1), ax=axes)
    # graph.figureplot(Try._equations.Freq,abs(Utip2), ax=axes)
    # print(abs(Utip))
    # Try.plottingWaveNumber()
    Try=Displacement_Field_FEM()
    Try.Hybrid_Displacement(isPlotting=True)
    # Try.constan_term()
    # plt.show()
 




