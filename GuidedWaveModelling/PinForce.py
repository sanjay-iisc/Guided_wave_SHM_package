
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

    my_dict_stress_parameter={'alpha_r': 1,'beta_r':0,'shearLag_r':0.9,
    'alpha_z': 1.17,'beta_z':0.41,'shearLag_z':0.92,'zeta':0}
    
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

        #-----Glue Properties
        self.hb=25e-6
        self.Eb = 3*(1+0.05j)*1e9
        self.vb=0.2
        
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
        t11= scipy.special.jv(self.alpha_r, self.shearLag_r* K * a) #scipy.special.jv(self.alpha_r, self.shearLag_r* K * a) * ((K/(1))  * a)**self.beta_r
        t22= self.zeta *  scipy.special.jv(self.alpha_z, self.shearLag_z*K *a) * ((K/(1))* a)**self.beta_z
        stressMatrix = np.array ([t11 , t22],dtype = complex)
        Mat_stress=stressMatrix.reshape((2,1))
        return Mat_stress
    
    # def S0_Stress_from_FEM(self,f):
    #     FemFreq = np.arange(5, 1000, 5)*1e3
    #     t11=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSR.npy"))
    #     t22=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSZ.npy"))
    #     stressMatrix = np.array ([t11(f) , t22(f)],dtype = complex)
    #     Mat_stress=stressMatrix.reshape((2,1))
    #     return Mat_stress
    # def A0_Stress_from_FEM(self,f):
    #     FemFreq = np.arange(5, 1000, 5)*1e3
    #     t11=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAR.npy"))
    #     t22=Spline(FemFreq ,np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAZ.npy"))
    #     stressMatrix = np.array ([t11(f) , t22(f)],dtype = complex)
    #     Mat_stress=stressMatrix.reshape((2,1))
    #     return Mat_stress   
        
    def constant_stress(self, alpha=4):
        
        Ep=1/self.s11e
        shai = (self.E*self.d)/ ((Ep*self.hp ))
        # print(shai)
        esia= (self.d31*self.Volt)/(self.hp) # strain piezo tip see paper [1]
        tau=( (shai)/(shai+alpha)) * (Ep) * (self.hp/self.a)*esia*self.a*self.a*2
        Eb = 3*(1+0.05j)
        vb=0.2
        hb=25e-6
        Gb= Eb/(2*(1+vb))
        Gamma= (abs((Gb/Ep) *(1/(self.hp*hb)) * ( (shai+alpha)/(shai))))

        # tau0=(Gb*esia)/(hb*Gamma*self.a**2)
        # tau=tau0*np.sinh(np.sqrt(Gamma)*self.a**2)*0.1
        # print(Gamma)
        # (shai+alpha)
        # Constant value of the pin force
        # alpha  = 4 for both modes, 1 is for symmetric and 3 is for antisym
        # Plate constants        
        # t = 1.5e-3# host thickness
        # nu =0.33 # plate
        # Ea = (1/ (1.60e-11)) # pizo elastic modulus
        # E = self.E #host elastic modulus

        # #Coupling constants
        # nuA= 0.33 # adhesive
        # Eb = 3e9 # Elastic moduluse of adhesive PA*(1+0.05j)*
        # tb = 25e-6 # bond thickness
        # nub = 0.3 # poisson ratio
        # Gb = Eb/ (2 *(1+nub))

        # # pzt constants
        # a=self.a
        # hp=self.hp
        # d31 = self.d31
        # V = 1 # volt
        
        # esia= (d31*V)/(hp) # strain piezo tip see paper [1]
        
        # shai = (E*t/  (1- nu**2))/ (Ea*hp/  (1- nuA**2) ) # Psi of equation 10 of [1]
                
        # shearLagA0 = np.sqrt( ( Gb / tb) * (  (1/ (Ea*hp) ) + (3/ (E * t))  )) # [2]
        # shearLagS0 =np.sqrt (( Gb / tb) * (  (1/ (Ea*hp) ) + (1/ (E * t))  )) # [2]
        
        # tau =( shai/ (shai+alpha)) * ( hp /a) * E*esia # [1] and following
        # tauS0 = tau* (1- 1/np.cosh(shearLagS0*a))
        # tauA0 = tau* (1- 1/np.cosh(shearLagA0*a))
        return (tau)#*self.a**2)
    def KapuriaModel(self, alpha=4):
        """"""
        Ga= self.Eb/(2*(1+self.vb))
        Ez=-1/self.hp
        d31=self.d31
        d32=self.d31
        Y1=1/self.s11e
        Y2=1/self.s11e
        s11=1/Y1
        s22=1/Y2
        v21=0.2
        v12=0.2
        s12=-v21/Y1
        Beta=self.hb/Ga
        s11_bar = s11-(s12**2/s22)
        d31_bar = self.d31- (d32*s12/s22)
        ###Elastic plates
        E1=self.E
        s11_s= 1/E1
        h =self.d*2

        tau1 = (alpha*s11_s)/(Beta*h)
        tau2 = (s11_bar)/(Beta*self.hp)

        tau =np.sqrt(tau1+tau2)

        x = np.linspace(-self.a , self.a , 1000, endpoint=True)
        sigma1 = (d31_bar*Ez*np.sinh(tau*x))

        sigma2 = Beta*tau*np.cosh(tau*self.a)

        
        sigma =(sigma1/sigma2)
        avg=np.sum(abs(sigma))/len(sigma) # avarage stresss
        # print(avg)
        return avg*self.a*self.a

class affective_radius:
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
        self._equations=WaveField()

    def AdmittanceRatio(self, isPlotting=False):
        ### Calculation of Slope value
        fff= np.linspace(300e3, 340e3,1000)
        ### Free Admittance Slope
        yf_pred,yf_des=self.Linear_Fit(fff, np.imag(affective_radius().Free_Admittance(fff) )
        ,self._equations.Freq)
        ### Bonded Admittance Slope
        y_pred,y_des=self.Linear_Fit(fff, np.imag(affective_radius.Bonded_Admittance(fff) )
        ,self._equations.Freq)
        print('Slope of Y {}'.format(y_des))
        print('Slope of Yf {}'.format(yf_des))
        a_effective=[]
        YY=affective_radius().Bonded_Admittance(self._equations.Freq)/affective_radius().Free_Admittance(self._equations.Freq)
        for i,ratio in enumerate(abs(YY)):
            if abs(ratio)>1:
                a_effective.append(1)
            else:
                a_effective.append(ratio)
        a_effective=np.array(a_effective)

        if isPlotting:
            fig,axes = plt.subplots(1,1, sharex=True)
            graph.figureplot(self._equations.Freq,abs(affective_radius().Bonded_Admittance(self._equations.Freq)),ax=axes, label='Bonded')
            graph.figureplot(self._equations.Freq,yf_pred,ax=axes, label='Unbonded-pred')
            graph.figureplot(self._equations.Freq,affective_radius().Free_Admittance(self._equations.Freq),ax=axes,label='Free')
            graph.figureplot(self._equations.Freq,abs(YY),ax=axes, label='Ratio')
        #     # fig,axes = plt.subplots(1,1, sharex=True)
        #     # graph.figureplot(t_w()._tipDisp._equations.Freq,abs(PRHW),ax=axes, title='PRHW')
        #     fig,axes = plt.subplots(1,1, sharex=True)
        #     graph.figureplot(t_w()._tipDisp._equations.Freq,abs(abs(YY)/abs(PRHW)),ax=axes, title='YY/PRHW')
        #     # graph.figureplot(t_w()._tipDisp._equations.Freq,abs(PRHW),ax=axes, title='YY/PRHW')
        return a_effective
    
    def Linear_Fit(self,X, Y,Xnew):
        X = X.reshape(-1, 1)  ### making 2d array here
        model = LinearRegression().fit(X, Y)
        r_sq = model.score(X, Y)
        Xnew = Xnew.reshape(-1, 1)
        print('coefficient of determination:', r_sq)
        # print('intercept:', new_model.intercept_)
        # print('slope:', new_model.coef_)
        y_pred = model.predict(Xnew)
        # print("predicted response:", y_pred, sep="\n")
        return y_pred, model.coef_[0]
    
    @staticmethod
    def plotting_measured_Admittance():
        fig,axes = plt.subplots(2,1, sharex=True)
        graph.figureplot(affective_radius()._equations.Freq,np.real(affective_radius().Bonded_Admittance(affective_radius()._equations.Freq))
        , ax=axes[0],
         xlabel='F[Hz]', ylabel='$|Y|[\Omega^{-1}]$', title ='Admittance curve for hp=125'+r'$\mu mm$',
         linestyle='None', marker='o', markersize=2, label ='Real-Bonded')
        graph.figureplot(affective_radius()._equations.Freq,np.imag(affective_radius().Bonded_Admittance(affective_radius()._equations.Freq)), 
        ax=axes[1],
         xlabel='F[Hz]', ylabel='$|Y|[\Omega^{-1}]$', title ='Admittance curve for hp=125'+r'$\mu mm$',
         linestyle='None', marker='o', markersize=2, label ='Imag-Bonded')
        #  graph.figureplot(affective_radius()._equations.Freq,np.imag(affective_radius().Bonded_Admittance(affective_radius()._equations.Freq)), 
        
         ### Unbonded
        graph.figureplot(affective_radius()._equations.Freq,np.real(affective_radius().Free_Admittance(affective_radius()._equations.Freq)), 
        ax=axes[0],
         xlabel='F[Hz]', ylabel='$|Y|[\Omega^{-1}]$', title ='Admittance curve for hp=125'+r'$\mu mm$',
         linestyle='None', marker='o', markersize=2, label ='Real-UnBonded')
        graph.figureplot(affective_radius()._equations.Freq,np.imag(affective_radius().Free_Admittance(affective_radius()._equations.Freq)), 
        ax=axes[1],
         xlabel='F[Hz]', ylabel='$|Y|[\Omega^{-1}]$', title ='Admittance curve for hp=125'+r'$\mu mm$',
         linestyle='None', marker='o', markersize=2, label ='Imag-Unbonded')



class Displacement_Field_PF:
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
            T_S=self._equations.Stress_function(ks,self._equations.a )# to check
            Dis_S1[i]=Const_S.dot((T_S))[0]*self._equations.KapuriaModel( alpha=1)#self._equations.constant_stress(alpha=1)
            Dis_S2[i]=Const_S.dot((T_S))[1]*self._equations.KapuriaModel( alpha=1)#self._equations.constant_stress(alpha=1)
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
            Dis_A1[i]=Const_A.dot((T_A))[0]*self._equations.KapuriaModel( alpha=3)
            Dis_A2[i]=Const_A.dot((T_A))[1]*self._equations.KapuriaModel( alpha=3)
        return Dis_A1,Dis_A2
    def PF_Displacement(self,indexS=[2],indexA=[1], isPlotting=False):
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
            fig,axes = plt.subplots(1,2, sharex=True)
            graph.figureplot(self._equations.Freq,abs((Urs*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[0], label='S0-PF',c='k')
            graph.figureplot(self._equations.Freq,abs((Ura*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[0], label='A0-PF',c='r')
            graph.figureplot(self._equations.Freq,Displacement_Field_PF.f_UrS0(self._equations.Freq) , ax=axes[0], label='S0-FEM',
            linestyle='None',marker='*',c='k', markersize=1)
            graph.figureplot(self._equations.Freq,Displacement_Field_PF.f_UrA0(self._equations.Freq) , ax=axes[0], label='A0-FEM',
            linestyle='None',marker='*',c='r', markersize=1, ylabel='Ur[mm]')

            #------------------------Uz----------------------------------
            graph.figureplot(self._equations.Freq,abs((Uzs*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[1], label='S0-PF',c='k')
            graph.figureplot(self._equations.Freq,abs((Uza*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[1], label='A0-PF',c='r')
            graph.figureplot(self._equations.Freq,Displacement_Field_PF.f_UzS0(self._equations.Freq) , ax=axes[1], label='S0-FEM',
            linestyle='None',marker='*',c='k', markersize=1)
            graph.figureplot(self._equations.Freq,Displacement_Field_PF.f_UzA0(self._equations.Freq) , ax=axes[1], label='A0-FEM',
            linestyle='None',marker='*',c='r', markersize=1, ylabel='Uz[mm]')
        
        return (Urs*np.pi*1j)/(2*self._equations.Mu),(Uzs*np.pi*1j)/(2*self._equations.Mu), 
        (Ura*np.pi*1j)/(2*self._equations.Mu),(Uza*np.pi*1j)/(2*self._equations.Mu)   # rad -m

class Displacement_Field_PF_effective_Radius:
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
        self._correction_radius=affective_radius().AdmittanceRatio()
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
            T_S=self._equations.Stress_function(ks,self._correction_radius[i]*self._equations.a )# to check
            Dis_S1[i]=((T_S))[0]*self._equations.KapuriaModel( alpha=1)#self._equations.constant_stress(alpha=1)Const_S.dot
            Dis_S2[i]=((T_S))[1]*self._equations.KapuriaModel( alpha=1)#self._equations.constant_stress(alpha=1)
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
            T_A=self._equations.Stress_function(ka,self._correction_radius[i]*self._equations.a )
            Dis_A1[i]=((T_A))[0]*self._equations.KapuriaModel( alpha=3)
            Dis_A2[i]=((T_A))[1]*self._equations.KapuriaModel( alpha=3)
        return Dis_A1,Dis_A2
    def PF_Displacement_with_effectiveRadius(self,indexS=[2],indexA=[1], isPlotting=False):
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
            fig,axes = plt.subplots(1,2, sharex=True)
            graph.figureplot(self._equations.Freq,abs((Urs*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[0], label='S0-PF',c='k')
            graph.figureplot(self._equations.Freq,abs((Ura*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[0], label='A0-PF',c='r')
            # graph.figureplot(self._equations.Freq,Displacement_Field_PF_effective_Radius.f_UrS0(self._equations.Freq) , ax=axes[0], label='S0-FEM',
            # linestyle='None',marker='*',c='k', markersize=1)
            # graph.figureplot(self._equations.Freq,Displacement_Field_PF_effective_Radius.f_UrA0(self._equations.Freq) , ax=axes[0], label='A0-FEM',
            # linestyle='None',marker='*',c='r', markersize=1, ylabel='Ur[mm]')

            #------------------------Uz----------------------------------
            graph.figureplot(self._equations.Freq,abs((Uzs*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[1], label='S0-PF',c='k')
            graph.figureplot(self._equations.Freq,abs((Uza*np.pi*1j)/(2*self._equations.Mu)) , ax=axes[1], label='A0-PF',c='r')
            graph.figureplot(self._equations.Freq,Displacement_Field_PF_effective_Radius.f_UzS0(self._equations.Freq) , ax=axes[1], label='S0-FEM',
            linestyle='None',marker='*',c='k', markersize=1)
            graph.figureplot(self._equations.Freq,Displacement_Field_PF_effective_Radius.f_UzA0(self._equations.Freq) , ax=axes[1], label='A0-FEM',
            linestyle='None',marker='*',c='r', markersize=1, ylabel='Uz[mm]')
        
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
    Try=Displacement_Field_PF()
    
    Try.PF_Displacement(isPlotting=True)
    # Try.constan_term()
    # Try=affective_radius()
    # Try.plotting_measured_Admittance()
    # Try.AdmittanceRatio(isPlotting=True)
    plt.show()
 




