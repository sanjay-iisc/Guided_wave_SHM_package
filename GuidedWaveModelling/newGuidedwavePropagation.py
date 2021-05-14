# -*- coding: utf-8 -*-

# References for equations
# [1] : Tuned Lamb wave excitation and detection with piezoelectric wafer active sensors for structural health monitoring 
# [2] : Lamb wave tuning curve calibration for surface-bonded piezoelectric transducers
# [3] : Hybrid empirical/analytical modeling of guided wave generation by circular piezoceramics 
# [4] : Book (giurgiutiu2007structural) page no 483 Giurgiutiu, V.Structural health monitoring: with piezoelectric wafer active sensors Elsevier, 2007
# %%
from __future__ import division
import sys
sys.path.append("./")
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
# plt.style.use('.\Plotting_style\science.mplstyle')
class WaveDisplacment:
    def __init__(self,isPlotting = False):

        # self.d   = (1.5/2) #mm: half thickness
        # self.E   = 70 # GPa
        # self.nu  = 0.33 # No unit
        # self.rho = 2.700 # g/cm3
        # self.a   = 5 # mm
        
        # self.K = np.load("K:\LMC\Sanjay\ForOlivier\\NicolasCode\WaveNumberMatrix.npy") #rad/mm
        # self.Freq = np.load("K:\LMC\Sanjay\ForOlivier\\NicolasCode\\Freq_WaveNumberMatrix.npy") #MHz
        
        # # self.K = np.load('E:\\Work\\Code\\SEFE\\wavenumbers.npy')[0][1][:,:29]
        # # self.Freq = np.load('E:\\Work\\Code\\SEFE\\freqs.npy')
        # self.omega=2*np.pi*self.Freq #rad.MHz
        # ### Model comsol for the nicolas
        # self.alpha_r    = 1.11
        # self.beta_r     = -0.22
        # self.alpha_z    = 1.17
        # self.beta_z     = 0.41
        # self.zeta       = 0.13
        # self.shearLag_r = 0.93
        # self.shearLag_z = 0.92
        # self.hp         = 0.125 # mm
        # # self.d31        = 175e-12 #m/V
        # self.d31        = -175e-9 #mm/V
        #-----------------------------
        self.d   = (1.5/2)*1e-3 #m: half thickness
        self.E   = 70e9 # GPa
        self.nu  = 0.33 # No unit
        self.rho = 2700 # kg/m3
        self.a   = 5e-3 # mm
        
        self.K= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix.npy")*1e3)[:-100, :]#rad/mm
        
        self.Freq = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix.npy")[:-100]*1e6#MHz
        
        self.ttt_ar=np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAR.npy")
        self.ttt_az=np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAZ.npy")
        self.ttt_sr=np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSR.npy")
        self.ttt_sz=np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSZ.npy")
        # self.K = np.load('E:\\Work\\Code\\SEFE\\wavenumbers.npy')[0][1][:,:50]*1e3
        # self.Freq = np.load('E:\\Work\\Code\\SEFE\\freqs.npy')*1e6
        self.omega=2*np.pi*self.Freq #rad/s
        ### Model comsol for the nicolas
        self.alpha_r    = 1.11
        self.beta_r     = -0.22
        self.alpha_z    = 1.17
        self.beta_z     = 0.41
        self.zeta       =0.13
        self.shearLag_r = 0.93
        self.shearLag_z = 0.92
        ### Observation point
        self.obs_r=25e-3 #mm observation point
        self.Nmodes = 49
        self.threshold = 1e-1
        # piezo Properties
        self.Volt = 1#10 ## voltage applied
        self.hp         = 0.125e-3 # mm
        self.d31        = -175e-12 #mm/V
        self.ns = 0.05 # damping value for PZT from 0 to 1
        # (N  / Vot^2) or Farades
        self.eps33=1790*8.85*1e-12*(1-1j*self.ns)#(N / Vot^2)#1900*e0#1800*e0# ## epsilon proper normlised
        self.s11e =16.4e-12*(1-1j*self.ns)#(M^2/N)#18e-12#1/(48*1e9)#18e-12 # 1/E11 for the 1d 
        self.rhoPiezo=7750#(#Kg/M^3)7800 #7600 ## density of PZT
        self.nu_p=0.35#piezo poisson ratio
        self.ComsolReultsPath="K:\LMC\Sanjay\Comsolresults\\NicolasResults\\NicolasResults4.csv"
        self.saveFigure='E:\PPT\Presentation\\02052021_ppt\Figure\\'
    def plottingWaveNumber(self): # for plotting the imported wavenumber 
        fig,axes = plt.subplots(1,1, sharex=True)
        for i in range(self.Nmodes):
            self.figureplot(self.Freq,self.K[:,i], ax=axes, xlabel='F[Hz]', ylabel='K[rad/m]', title ='Wave Number-1.5mmplate',linestyle='None', marker='o', markersize=2,c='k')
            self.figureplot(self.Freq,np.imag(self.K[:,i]), ax=axes, xlabel='F[Hz]', ylabel='K[rad/m]', title ='Wave Number-1.5mmplate',linestyle='None', marker='o', markersize=2,c='r')     
        # self.figureplot(self.Freq,np.imag(AD),ax=axes[1], xlabel='F[Hz]', ylabel='Imag[Ad][Siemens]', label='Unbonded-Ana',title ='Admittance Curve')

    def freePZTimpedance(self, isPlotting=True):
        # ref. [4] page 483
        A = np.pi* self.a**2 ## area of PZT
        E3 = -self.Volt/self.hp ## Electric felid in the PZT
        ## equation constent 
        c =np.sqrt(1/ ( self.s11e*self.rhoPiezo *(1-self.nu_p**2) )) #np.sqrt(1/ ( s11e*rho )) ## wave velocity
        SISA = self.d31 * E3### Strain induced at the tip of the PZT
        UISA = SISA * self.a ### Displacement at the tip of the PZT
        k = self.omega/c ## omega / c here w is omega not frequency
        K31 =self.d31**2 / (self.s11e* self.eps33) ## coupling coefficient of the PZT
        KP = np.sqrt( (2 * K31) / (1-self.nu_p))
        C = ( (self.eps33) *A)/self.hp ## capacitance of PZT# page 66 
        # print(C)
        phi = k*self.a
        ### Equation 
        #current
        ## edge displacement 
        ur_a1 = (1+self.nu_p) * jv(1, phi)
        ur_a2 =  phi * jv(0, phi) -(1-self.nu_p)*jv(1,phi)
        ur_a = ur_a1/ ur_a2
        ## Charge
        Q = C  *(  1 - KP**2 * (1 - ur_a )) * self.Volt 
        # Current 
        I = 1j*self.omega*Q
        print('capacitance =',C)
        # Admittance
        AD = I / self.Volt
        ## Importing free PZT admittance for the 5mm radius and self.hp =0.125 mm PZT from COMSOL to compared the results
        FreeImFile = pd.read_csv("K:\LMC\Sanjay\Comsolresults\\NicolasResults\\NicolasResults3_freeImpedance.csv", skiprows=4)
        freeFreq=FreeImFile['% freq (kHz)']*1e3
        freeIm=(FreeImFile['i*( es.nD*es.omega) (A), Boundary Probe 1']).str.replace('i','j').apply(lambda x: np.complex128(x))
        f_freeIm= interp1d(freeFreq,freeIm)
        if isPlotting:
            fig,axes = plt.subplots(2,1, sharex=True)
            self.figureplot(self.Freq,np.real(AD), ax=axes[0], xlabel='F[Hz]', ylabel='Real[Ad][Siemens]', label='Unbounded-Ana', title ='Admittance Curve')
            self.figureplot(self.Freq,np.imag(AD),ax=axes[1], xlabel='F[Hz]', ylabel='Imag[Ad][Siemens]', label='Unbounded-Ana',title ='Admittance Curve')
            #### ---->>> FEM results
            self.figureplot(freeFreq,np.real(freeIm), ax=axes[0],marker='o',markersize=0.9, c='k', linestyle ='None',xlabel='F[Hz]', ylabel='Real[Ad][Siemens]', label='Unbonded-FEM', title ='Admittance Curve')
            self.figureplot(freeFreq,np.imag(freeIm),ax=axes[1],marker= 'o',linestyle ='None',markersize=0.9, c='k',xlabel='F[Hz]', ylabel='Imag[Ad][Siemens]', label='Unbonded-FEM',title ='Admittance Curve')
        return f_freeIm(self.Freq)
    ### comsol Results Importing form the FEM
    def COMSOLresults(self):
        dis = pd.read_csv(self.ComsolReultsPath, skiprows=4)
        freqFEM=dis['freq (kHz)'].to_numpy()*1e3 #in Hz
        current=(dis['Current (A), Boundary Probe 1']).str.replace('i','j').apply(lambda x: np.complex128(x)) # at 1 volt
        bonded_AD=current/self.Volt # I/V for Admittance
        fBondedImpedance=interp1d(freqFEM,bonded_AD) # interpolating to make on the same axis for Impedance
        #-----Free Impedance -------
        ## Importing free PZT admittance for the 5mm radius and self.hp =0.125 mm PZT from COMSOL to compared the results
        FreeImFile = pd.read_csv("K:\LMC\Sanjay\Comsolresults\\NicolasResults\\NicolasResults3_freeImpedance.csv", skiprows=4)
        freeFreq=FreeImFile['% freq (kHz)']*1e3
        freeIm=(FreeImFile['i*( es.nD*es.omega) (A), Boundary Probe 1']).str.replace('i','j').apply(lambda x: np.complex128(x))
        f_freeIm= interp1d(freeFreq,freeIm)
        ###%%%% Tip displacement from the Comsol
        UtipFEm=dis['Displacement field, R component (mm), Point: (5, 1.5)'].str.replace('i','j').apply(lambda x: np.complex128(x))
        #-----Normalized with respect to d31*V*a/ta------
        UtipFEm=(UtipFEm)#*self.hp )/(self.d31*self.Volt*self.a)
        f_UtipFEm=interp1d(freqFEM,UtipFEm)
        #---------Displacement--------
        UrS0=dis['S0_u']*1e-3
        UrA0=dis['A0_u']*1e-3
        f_UrS0=interp1d(freqFEM,UrS0)
        f_UrA0=interp1d(freqFEM,UrA0)
        Data= {'bonded_AD':fBondedImpedance(self.Freq),'Ur_tip[m]':f_UtipFEm(self.Freq), 'freeAd':f_freeIm(self.Freq),'UrS0':f_UrS0(self.Freq)
        ,'UrA0':f_UrA0(self.Freq)  }
        return Data

    def BulkwaveSpeed(self):
        self.Lambada = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))

        self.Mu = self.E / (2 * (1 + self.nu))

        self.C_L = math.sqrt((self.Lambada + 2 * self.Mu) / self.rho)

        self.C_T = math.sqrt(self.Mu / self.rho)
        # print(self.C_T)
        return self.C_L, self.C_T
    
    def da_dash(self, K , W ):
        # Derivative of equ 20 of [3]
        self.BulkwaveSpeed()
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
        self.BulkwaveSpeed()
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
        self.BulkwaveSpeed()
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
        self.BulkwaveSpeed()
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
        self.BulkwaveSpeed()
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
        
        self.Ns = np.array( [[N11 , N12 ], [N21 , N22 ]], dtype =complex)
        
        return self.Ns
    
    def N_Antisym(self, K , W, angle = np.deg2rad(0)):
        '''
        Args : K = waveNumber
        a = matrix 
        '''
        self.BulkwaveSpeed()
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
        
        ## 
        
        self.Na = np.array( [[N11 , N12 ], [N21 , N22 ]], dtype =complex)
        
        return self.Na
    
    def HankelMatrix (self , K, r):
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
    
    def HankelMatrix_2 (self , K, r):
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
    
    def constant_stress(self, alpha=4):
        # Constant value of the pin force
        # alpha  = 4 for both modes, 1 is for symmetric and 3 is for antisym
        # Plate constants        
        t = 1.5e-3# host thickness
        nu =0.33 # plate
        Ea = (1/ (1.60e-11)) # pizo elastic modulus
        E = self.E #host elastic modulus

        #Coupling constants
        nuA= 0.33 # adhesive
        Eb = 3e9 # Elastic moduluse of adhesive PA*(1+0.05j)*
        tb = 25e-6 # bond thickness
        nub = 0.3 # poisson ratio
        Gb = Eb/ (2 *(1+nub))

        # pzt constants
        a=self.a
        hp=self.hp
        d31 = self.d31
        V = 1 # volt
        
        esia= (d31*V)/(hp) # strain piezo tip see paper [1]
        
        shai = (E*t/  (1- nu**2))/ (Ea*hp/  (1- nuA**2) ) # Psi of equation 10 of [1]
                
        shearLagA0 = np.sqrt( ( Gb / tb) * (  (1/ (Ea*hp) ) + (3/ (E * t))  )) # [2]
        shearLagS0 =np.sqrt (( Gb / tb) * (  (1/ (Ea*hp) ) + (1/ (E * t))  )) # [2]
        
        tau =( shai/ (shai+alpha)) * ( hp /a) * E*esia # [1] and following
        tauS0 = tau* (1- 1/np.cosh(shearLagS0*a))
        tauA0 = tau* (1- 1/np.cosh(shearLagA0*a))
        return abs(tauS0), abs(tauA0)
    
    
    def Stress_function(self, K, a ):
        # right hand side of vector of equation 27 of [3]
        '''
        K = Wave number in rad / meter
        a = Radius of Piezo in m
    
        '''
        # A,B=self.constant_stress()
        #---- The unit for this function is weird so we have to change the K unit in (1/m) from (rad / m) because the function will change is (K*a)
        # power of 0.2 so best to change the unit here!
        #---- K / 2*np.pi----
       
        t11= scipy.special.jv(self.alpha_r, self.shearLag_r* K * a) * (K * a)**self.beta_r
        t22= self.zeta *  scipy.special.jv(self.alpha_z, self.shearLag_z*K *a) * (K* a)**self.beta_z
    
        stressMatrix = np.array ([t11 , t22],dtype = complex)
        Mat_stress=stressMatrix.reshape((2,1))
        
        return Mat_stress
    
    
   
    def pinForcestress(self,K,a):
        
        # print('Codd',A)
        t11= scipy.special.jv(1, self.shearLag_r*K * a)/a
        t22=0#A*self.a*scipy.special.jv(0, K * a)
        stressMatrix = np.array ([t11 , t22],dtype = complex)
        Mat_stress=stressMatrix.reshape((2,1))
        
        return Mat_stress

    def symDisplacmentPRHW(self,k):
        """ if is stress is 1 then stress multiply by hp else 0 it is withouth hp multiply"""
        Dis_S1=np.zeros_like(self.Freq, dtype=complex)
        Dis_S2=np.zeros_like(self.Freq, dtype=complex)
        #ur and uz
        for i, ks in enumerate(k):
            dss=self.ds_dash(ks, self.omega[i])#scalar ; denominator of eq 23 of [3]
            Ns=self.N_sym( ks ,self.omega[i]) # matrix 2*2; numerator of eq 23 of [3]
            Amp_SS = (Ns)*(ks/dss) # Matrix 2*2 ; fraction of equ 23
            Hs=self.HankelMatrix(ks,self.a) # Matrix 2*2 # Eq 23
            Const_S = np.matmul(Hs,Amp_SS )
            T_S=self.Stress_function(ks,self.a )# to check
            Dis_S1[i]=Const_S.dot((T_S))[0]
            Dis_S2[i]=Const_S.dot((T_S))[1]
        return Dis_S1,Dis_S2
            
    
    def antiDisplacmentPRHW(self,k):
        """ if is stress is 0 then stress multiply by hp else 1 it is withouth hp multiply"""
        Dis_A1=np.zeros_like(self.Freq, dtype=complex)
        Dis_A2=np.zeros_like(self.Freq, dtype=complex)
        for i, ka in enumerate(k):
            daa=self.da_dash(ka, self.omega[i])#one value
            Na=self.N_Antisym( ka ,self.omega[i]) # matrix 2*2
            Amp_AA = (Na)*(ka/daa) # Matrix 2*2
            Ha=self.HankelMatrix(ka,self.a)
            Const_A = np.matmul(Ha,Amp_AA )
            T_A=self.Stress_function(ka,self.a )
            Dis_A1[i]=Const_A.dot((T_A))[0]
            Dis_A2[i]=Const_A.dot((T_A))[1]
        return Dis_A1,Dis_A2
    
    #---------------------actual calculations--------------------------
    def symDisplacment(self,k,r):
        """ if is stress is 1 then stress multiply by hp else 0 it is withouth hp multiply"""
        Dis_S1=np.zeros_like(self.Freq, dtype=complex)
        Dis_S2=np.zeros_like(self.Freq, dtype=complex)
        #####----Freq --dependent
        TW=self.stressComparision_bw_FEM()
        
        #ur and uz
        for i, ks in enumerate(k):
            dss=self.ds_dash(ks, self.omega[i])#scalar ; denominator of eq 23 of [3]
            Ns=self.N_sym( ks ,self.omega[i]) # matrix 2*2; numerator of eq 23 of [3]
            Amp_SS = (Ns)*(ks/dss) # Matrix 2*2 ; fraction of equ 23
            Hs=self.HankelMatrix_2(ks,r) # Matrix 2*2 # Eq 23
            Const_S = np.matmul(Hs,Amp_SS )
            T_S_11,T_S_22=self.Stress_function(ks,self.a )# to check
            T_S = np.array ([T_S_11*(TW[i])*self.a , T_S_22*self.hp*(TW[i])*self.a],dtype = complex)
            T_S=T_S.reshape((2,1))
            Dis_S1[i]=Const_S.dot((T_S))[0]
            Dis_S2[i]=Const_S.dot((T_S))[1]
        return Dis_S1,Dis_S2
            
    
    def antiDisplacment(self,k,r):
        """ if is stress is 0 then stress multiply by hp else 1 it is withouth hp multiply"""
        Dis_A1=np.zeros_like(self.Freq, dtype=complex)
        Dis_A2=np.zeros_like(self.Freq, dtype=complex)
        #####----Freq --dependent
        TW=self.stressComparision_bw_FEM()
        
        for i, ka in enumerate(k):
            daa=self.da_dash(ka, self.omega[i])#one value
            Na=self.N_Antisym( ka ,self.omega[i]) # matrix 2*2
            Amp_AA = (Na)*(ka/daa) # Matrix 2*2
            Ha=self.HankelMatrix_2(ka,r)
            Const_A = np.matmul(Ha,Amp_AA )
            T_A_11,T_A_22=self.Stress_function(ka,self.a )# to check
            T_A = np.array ([T_A_11*(TW[i])*self.a , T_A_22*self.hp*(TW[i])*self.a],dtype = complex)
            T_A=T_A.reshape((2,1))
            Dis_A1[i]=Const_A.dot((T_A))[0]
            Dis_A2[i]=Const_A.dot((T_A))[1]
        return Dis_A1,Dis_A2

    def Displacement_calulation(self, isPlotting=False):
        KS= self.K[:,2] # Wave number for S0 n=2
        KA= self.K[:,1] # Wave number for A0 n=2
        US=self.symDisplacment(KS,self.obs_r)
        UA=self.antiDisplacment(KA,self.obs_r)
        Data=self.COMSOLresults()
        if isPlotting:
            fig, axes=plt.subplots(1,1,sharex=True)
            self.figureplot(self.Freq, abs(US[0]), ax=axes, label = 'S0')
            self.figureplot(self.Freq, abs(UA[0]), ax=axes, label = 'A0')
            self.figureplot(self.Freq, Data['UrS0']*1e-3, ax=axes, label = 'S0-FEM')
            self.figureplot(self.Freq, Data['UrA0']*1e-3, ax=axes, label = 'A0-FEM')


    def PRHW(self,isPlotting=False, isSavefig=False):
        # eq 30 of [3]
        # eq 30 of [3]
        Da= np.zeros_like(self.Freq, dtype=complex)
        Ds= np.zeros_like(self.Freq, dtype=complex)
        DA= np.zeros((len(self.Freq),self.Nmodes),  dtype=complex)
        DS= np.zeros((len(self.Freq),self.Nmodes),  dtype=complex)
        temp=0#np.zeros_like(self.Freq, dtype=complex)
        tempz=0
        
        for n in np.arange(0,self.Nmodes,1):
            k=self.K[:,n]
            Da=self.da_equation((k), self.omega)
            Ds=self.ds_equation((k), self.omega)
            if max(abs(Ds))< self.threshold :
                print('We have found a S mode!',n)
                Dis_Sr,Dis_Sz=self.symDisplacmentPRHW(k)
                temp=temp+(Dis_Sr)
                tempz=tempz+(Dis_Sz)
            if max(abs(Da))< self.threshold:
                print('We have found a A mode!',n)
                Dis_Ar,Dis_Az=self.antiDisplacmentPRHW(k)
                temp=temp+(Dis_Ar)
                tempz=tempz+(Dis_Az)
            if max(abs(Ds))< self.threshold and max(abs(Da))< self.threshold:
                print('warning')
        Data=self.COMSOLresults()
        TipDisp=(abs(temp)+abs(tempz))/2
        #----- PRHW-uint is  [rad/m] so to normalised this (a/2pi) becaus the t= (t/2mue)
        if isPlotting:
            #+(np.abs(tempz)*(self.a/ (4*np.pi)))
            fig, axes=plt.subplots(1,1,sharex=True)
            self.figureplot(self.Freq, TipDisp, ax=axes, title='total displacment at tip',label= '$U_r$-Ana',linestyle='None' ,marker='o',markersize=2, c='k',xlabel='F[Hz]', ylabel=r'[$\dfrac{a}{2\pi}$]-Normalised')
            # self.figureplot(self.Freq, np.abs(tempz), ax=axes, title='total displacment at tip',label= '$U_z$', marker='o',markersize=2, c='r',xlabel='F[Hz]')
            # self.figureplot(self.Freq, ((np.abs(Data['Ur_tip[m]']))), ax=axes, title='total displacment at tip',label= r'$U_r$-FEM',linestyle='None' ,marker='o',markersize=2, c='r',xlabel='F[Hz]', ylabel=r'[$\dfrac{a}{2\pi}$]-Normalised')
            # fig, axes=plt.subplots(1,1,sharex=True)
            # self.figureplot(self.Freq, abs((np.abs(Data['Ur_tip[m]']))-(np.abs(temp)*(self.a/ (2*np.pi)))), ax=axes, label='A'+str(n))
            # self.figureplot(self.Freq, np.abs(Ds), ax=axes, label='S'+str(n))
        np.save("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\\Utip.npy",TipDisp)#+abs(tempz)/ (2*np.pi))/2)
        return TipDisp#/2# ax=axes, label='S'+str(i), linestyle='None', marker ='*', c='r')
            
    
    
    def tw_term(self,isPlotting=False, isSavefig=False):
        Data=self.COMSOLresults() ### Importing from the COMSOL
        y = (Data['bonded_AD'])
        yf=(Data['freeAd'])#(self.freePZTimpedance())'freeAd'
        ####----Linear Fit
        fff= np.linspace(10e3, 100e3,1000)
        yf_inter=self.inter_fun(self.Freq, yf, fff)
        yf_pred,model_para=self.Linear_Fit(fff, np.imag(yf_inter),self.Freq)
        print('model_pra=',model_para)
        #%%%%%--- y' and yf' at omega = 0
        ##--- forword differnce formula (f(a+h)-f(a))/h at w= 0 but the starting is 10 khz
        dw = self.omega[1]-self.omega[0]
        y_desh = (y[1]-y[0])/dw
        yf_desh = (yf[1]-yf[0])/dw
        print('y_desh={}'.format(y_desh))
        print('yf_desh={}'.format(yf_desh))
        ###%%%%%%-----> (y(w)-w*y'(0))/(w*(yf'(0)-y'(0))
        Adterm1 = y-self.omega*y_desh
        Adterm2 = self.omega*(yf_desh-y_desh)
        Adterm= Adterm1/Adterm2#yf_pred#np.imag(y)/yf_pred#
        ###-- Calling the function for PRHW
        # Utip =self.PRHW()
        Tw= np.abs(Adterm)*self.d31*self.a
        utip=np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\\Utip.npy")
        if isPlotting:
            fig, axes=plt.subplots(1,1,sharex=True)
            self.figureplot(self.Freq, ((Tw)), ax=axes, label='Ad', title='Tw-Terms', linestyle='None', marker='*')
            self.figureplot(self.Freq, (np.abs(utip)), ax=axes, label='PRHW', title='Tw-Terms', linestyle='None', marker='*')
            # self.figureplot(self.Freq, ((Tw))/(np.abs(utip)*self.a), ax=axes, label='PRHW', title='Tw-Terms', linestyle='None', marker='*')
        #     fig, axes=plt.subplots(1,1,sharex=True)
        #     self.figureplot(self.Freq, 1/np.abs(Utip), ax=axes, label='Utip-Ana', title='Comparison of PRHW and Admittance')
        #     self.figureplot(self.Freq, np.abs(Adterm), ax=axes, label='Admittance', title='Comparison of PRHW and Admittance')
            fig, axes=plt.subplots(1,1,sharex=True)
            self.figureplot(self.Freq, np.imag(y), ax=axes, label='Bonded', title='Admittance')
            self.figureplot(self.Freq, np.imag(yf), ax=axes, label='Free', title='Admittance')
            self.figureplot(self.Freq, yf_pred, ax=axes, label='Free-fit', title='Admittance')
        #     fig, axes=plt.subplots(1,1,sharex=True)
        #     self.figureplot(self.Freq, yf_pred/np.imag(y), ax=axes, label='y/y_fit', title='Admittance')
        #     # self.figureplot(self.Freq, np.abs(Ds), ax=axes, label='S'+str(n))
        return Tw
    def frequency_dependent_stress(self,isPlotting=False, isSavefig=False):
        Tw=self.tw_term()
        KS= self.K[:,2] # Wave number for S0 n=2
        KA= self.K[:,1] # Wave number for A0 n=2
        StressS0 =np.zeros((len(KS),2),dtype=complex)
        StressA0 =np.zeros((len(KS),2),dtype=complex)
        const_stress= np.array ([1 , self.hp]).reshape((2,1))
        for i in  range(len(Tw)):
            StressS0[i,:]=np.multiply(self.Stress_function(KS[i]),const_stress).T * Tw[i]*self.a
            StressA0[i,:]=np.multiply(self.Stress_function(KA[i]),const_stress).T* Tw[i]*self.a
        
        if isPlotting:
            fig, axes=plt.subplots(1,1,sharex=True)
            self.figureplot(self.Freq, np.abs(StressS0[:,0]), ax=axes, label='S0', title='$\sigma_{rr}$')
            self.figureplot(self.Freq, np.abs(StressA0[:,0]), ax=axes, label='A0', title='$\sigma_{rr}$')
    #### For checking the equations
    def checkTheequation(self):
        KS= self.K[:,2] # Wave number for S0 n=2
        KA= self.K[:,1] # Wave number for A0 n=2

        ##%%% N
        NS0 = np.zeros_like(KS, dtype=complex)
        NA0 = np.zeros_like(KA, dtype=complex)
        ##%%% D_desh
        DS0 = np.zeros_like(KS, dtype=complex)
        DA0 = np.zeros_like(KA, dtype=complex)

        for i in range(len(self.Freq)):
            DS0[i]=self.ds_dash(KS[i], self.omega[i])#scalar ; denominator of eq 23 of [3]
            NS0[i]=self.N_sym( KS[i], self.omega[i]) [0][0]*KS[i]
            DA0[i]=self.da_dash(KA[i], self.omega[i])#one value
            NA0[i]=self.N_Antisym( KA[i] ,self.omega[i])[0][0]*KA[i] # matrix 2*2
        fig, axes=plt.subplots(1,1,sharex=True)
        # self.figureplot(self.Freq, np.abs(NS0/DS0), ax=axes, label='$NS/DS$')
        # self.figureplot(self.Freq, np.abs(NA0/DA0), ax=axes, label='$NA/DA$', title='Comaprision')
        self.figureplot(self.Freq, np.abs(NA0/DA0)/np.abs(NS0/DS0), ax=axes, ylabel='$(NA/DA)/(NS/DS)$', title='Ratio of stiffnes',linestyle='--')
    
    def checkRootsTheequation(self):
        KS= self.K[:,2] # Wave number for S0 n=2
        KA= self.K[:,1] # Wave number for A0 n=2

        ##%%% N
        NS0 = np.zeros_like(KS, dtype=complex)
        NA0 = np.zeros_like(KA, dtype=complex)
        ##%%% D_desh
        DS0 = np.zeros_like(KS, dtype=complex)
        DA0 = np.zeros_like(KA, dtype=complex)

        for i in range(len(self.Freq)):
            DS0[i]=self.ds_equation(KS[i]*self.d, self.omega[i]*self.d)#scalar ; denominator of eq 23 of [3]
            DA0[i]=self.da_equation(KA[i]*self.d, self.omega[i]*self.d)#one value
           
        fig, axes=plt.subplots(2,1,sharex=True)
        # self.figureplot(self.Freq, np.abs(NS0/DS0), ax=axes, label='$NS/DS$')
        # self.figureplot(self.Freq, np.abs(NA0/DA0), ax=axes, label='$NA/DA$', title='Comaprision')
        self.figureplot(self.Freq, np.abs(DA0), ax=axes[0], ylabel='$DA$', title='Ratio of stiffnes',linestyle='--')
        self.figureplot(self.Freq, np.abs(DS0), ax=axes[1], ylabel='$DS$', title='Ratio of stiffnes',linestyle='--')
    
    def stressComparision_bw_FEM(self,isPlotting=False):
        self.BulkwaveSpeed()
        KS= self.K[:,2] # Wave number for S0 n=2
        KA= self.K[:,1] # Wave number for A0 n=1
        # self.optimum_radiusA()
        # self.optimum_radiusS()
        ttt_SR = np.zeros_like(KS)
        ttt_AR = np.zeros_like(KA)
        ttt_SZ = np.zeros_like(KS)
        ttt_AZ = np.zeros_like(KA)
        for i in range(len(self.Freq)):
            T_S=self.Stress_function(KS[i], self.a)
            T_A=self.Stress_function(KA[i] ,self.a)
            ttt_SR[i]=T_S[0]*self.a # Multiply * 2*mue and divided by pi b
            #bcoz Residul theorms says 2*pi and change in the integration from -inf to inf thats why divide by 2 
            ttt_SZ[i]=T_S[1]*self.a*self.hp
            ttt_AR[i]=T_A[0]*self.a
            ttt_AZ[i]=T_A[1]*self.a*self.hp
        ### interpolating the 
        freqFem = np.arange(5, 1000, 5)
        fff_femAR = interp1d(freqFem*1e3, self.ttt_ar)
        ratioAR= fff_femAR(self.Freq)/(ttt_AR)
        fff_femSR = interp1d(freqFem*1e3, self.ttt_sr)
        fff_femSZ = interp1d(freqFem*1e3, self.ttt_sz)
        fff_femAZ = interp1d(freqFem*1e3, self.ttt_az)
        ratioSR= fff_femSR(self.Freq)/abs(ttt_SR)
        All_total=(abs(ttt_SR+ttt_AR+ttt_SZ+ttt_AZ))
        ratioT_Sr = (fff_femSR(self.Freq))/ (abs(ttt_SR+ttt_AR))
        ratioT_Ar = (fff_femAR(self.Freq))/ (abs(ttt_SR+ttt_AR))
        ratioAllT_Sr = (fff_femSR(self.Freq)+fff_femSZ(self.Freq))/All_total 
        ratioAllT_Ar = (fff_femAR(self.Freq)+fff_femAZ(self.Freq))/All_total
        # +fff_femAR(self.Freq)
        Tw=abs(ratioAllT_Ar+ratioAllT_Sr)#abs(ratioT_Ar+ratioT_Sr)/2
        if isPlotting:
            
            ### interpolating the 
            fff_femAR = interp1d(freqFem*1e3, (self.ttt_ar))
            
            # newRatio =new_ttt_ar/new_ttt_AR
            #####
            fig, axes=plt.subplots(1,1)
            self.figureplot(self.Freq, abs(fff_femAR(self.Freq)), ax=axes, ylabel='$\sigma_{rr}[Pa-m^2]$', title='Comparison of FEM and Hybrid Interfacial '+r'$\sigma_{rr}$'+'-A0 mode', label ='FEM', linestyle='None', marker='o', markersize=2, c='b', filename='sigma_rrA0')
            self.figureplot(self.Freq, abs((ttt_AR)), ax=axes ,ylabel='$\sigma_{rr}$', title='Comparison of FEM and Hybrid Interfacial '+r'$\sigma_{rr}$'+'-A0 mode',linestyle='-',marker='None', c='k', label ='Hybrid',markersize=2, filename='sigma_rrA0')
            
            # # self.figureplot(newFreqfem,new_ttt_ar, ax=axes[1], ylabel='$\tau_{rr}$', title='comparision of FEM and Analytical',linestyle='None',marker='o', c='k', label ='A0-Ana')
            # # self.figureplot(newFreq,new_ttt_AR, ax=axes[1], ylabel='$\tau_{rr}$', title='comparision of FEM and Analytical', label ='A0-FEM', linestyle='None', marker='o', markersize=1, c='b')
            fig, axes=plt.subplots(1,1)
            self.figureplot(self.Freq, abs(fff_femSR(self.Freq)), ax=axes, ylabel='$\sigma_{rr}[Pa-m^2]$', title='Comparison of FEM and Hybrid Interfacial '+r'$\sigma_{rr}$'+'-S0 mode', label ='FEM', linestyle='None', marker='o', markersize=2, c='b', filename='sigma_rrS0')
            self.figureplot(self.Freq, abs(ttt_SR), ax=axes, ylabel='$\sigma_{rr}$', title='Comparison of FEM and Hybrid Interfacial '+r'$\sigma_{rr}$'+'-S0 mode',linestyle='-', c='k', label ='Hybrid',marker='None', markersize=2,filename='sigma_rrS0')
            

            fig, axes=plt.subplots(1,1)
            self.figureplot(self.Freq, abs(fff_femAZ(self.Freq)), ax=axes, ylabel='$\sigma_{zz}[Pa-m^2]$', title='Comparison of FEM and Hybrid Interfacial '+r'$\sigma_{zz}$'+'-A0 mode', label ='FEM', linestyle='None', marker='o', markersize=2, c='b', filename='sigma_zzA0')
            self.figureplot(self.Freq, abs((ttt_AZ)), ax=axes ,ylabel='$\sigma_{zz}$', title='Comparison of FEM and Hybrid Interfacial '+r'$\sigma_{zz}$'+'-A0 mode',linestyle='-',marker='None', c='k', label ='Hybrid',markersize=2, filename='sigma_zzA0')
            
            # # self.figureplot(newFreqfem,new_ttt_ar, ax=axes[1], ylabel='$\tau_{rr}$', title='comparision of FEM and Analytical',linestyle='None',marker='o', c='k', label ='A0-Ana')
            # # self.figureplot(newFreq,new_ttt_AR, ax=axes[1], ylabel='$\tau_{rr}$', title='comparision of FEM and Analytical', label ='A0-FEM', linestyle='None', marker='o', markersize=1, c='b')
            fig, axes=plt.subplots(1,1)
            self.figureplot(self.Freq, abs(fff_femSZ(self.Freq)), ax=axes, ylabel='$\sigma_{zz}[Pa-m^2]$', title='Comparison of FEM and Hybrid Interfacial '+r'$\sigma_{zz}$'+'-S0 mode', label ='FEM', linestyle='None', marker='o', markersize=2, c='b', filename='sigma_zzS0')
            self.figureplot(self.Freq, abs(ttt_SZ), ax=axes, ylabel='$\sigma_{zz}$', title='Comparison of FEM and Hybrid Interfacial '+r'$\sigma_{zz}$'+'-S0 mode',linestyle='-', c='k', label ='Hybrid',marker='None', markersize=2,filename='sigma_zzS0')
            

            
            fig, axes=plt.subplots(1,1,sharex=True, sharey=False)
            self.figureplot(self.Freq, abs(ratioAR), ax=axes,ylim=[0,2e2] ,ylabel=r'$\frac{\tau^{FEM}_{rz}}{\tau^{Ana}_{rz}}$', title=r'$\tau(\omega)$'+'-A0',linestyle='-', markersize=1, c='k')
            fig, axes=plt.subplots(1,1,sharex=True, sharey=False)
            self.figureplot(self.Freq, abs(ratioSR), ax=axes,ylim=[0,2e2], ylabel=r'$\frac{\tau^{FEM}_{rz}}{\tau^{Ana}_{rz}}$', title=r'$\tau(\omega)$'+'-S0',linestyle='-', markersize=1, c='r')
            fig, axes=plt.subplots(1,1,sharex=True, sharey=False)
            self.figureplot(self.Freq, abs(ratioT_Sr), ax=axes,ylim=[0,2e2], ylabel=r'$\frac{\tau^{FEM}_{rz}}{\tau^{A_Tot}_{rz}}$', title=r'$\tau(\omega)$'+'-S0',linestyle='-', markersize=1, c='r')
            fig, axes=plt.subplots(1,1,sharex=True, sharey=False)
            self.figureplot(self.Freq, abs(ratioT_Ar), ax=axes,ylim=[0,2e2], ylabel=r'$\frac{\tau^{FEM}_{rz}}{\tau^{A_Tot}_{rz}}$', title=r'$\tau(\omega)$'+'-A0',linestyle='-', markersize=1, c='r')
            fig, axes=plt.subplots(1,1)
            self.figureplot(self.Freq,abs(Tw) , ax=axes,ylim=[0,2e2], ylabel=r'$\frac{\tau_{effective}(k,\omega)}{\tau_{effective}(k)}}$'+'[N-m]', title=r'$\tau(\omega)$',linestyle='-', markersize=2, c='r', filename='tau_omega')
            fig, axes=plt.subplots(1,1,sharex=True, sharey=False)
            self.figureplot(self.Freq,abs(Tw*ttt_SR) , ax=axes, ylabel=r'$\tau_{rz}$', title=r'$\tau^{Ana}_{rz}$'+'-S0',linestyle='-', markersize=1, c='r')
            fig, axes=plt.subplots(1,1,sharex=True, sharey=False)
            self.figureplot(self.Freq,abs(Tw*ttt_AR) , ax=axes, ylabel=r'$\tau_{rz}$', title=r'$\tau^{Ana}_{rz}$'+'-A0',linestyle='-', markersize=1, c='r')
            
        return Tw
    
   
    def stressFunction_tooptimize(self):
        KS= self.K[:,2] # Wave number for S0 n=2
        KA= self.K[:,1] # Wave number for A0 n=2
        sigma_Ar= np.zeros_like(self.Freq, dtype=complex)
        sigma_Sr= np.zeros_like(self.Freq, dtype=complex)
        TW=self.tw_term(isPlotting=True)
        utip=np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\\Utip.npy")
        for i,(ka,ks) in enumerate(zip(KA,KS)):
            T_A=self.Stress_function_2(ka ,self.a)
            T_S=self.Stress_function_2(ks ,self.a)
            sigma_Ar[i]=T_A[0]*(TW[i]/utip[i])*self.a
            sigma_Sr[i]=T_S[0]*(TW[i]/utip[i])*self.a
        plt.figure()
        plt.plot(self.Freq, abs(sigma_Ar))
        plt.plot(self.Freq, abs(sigma_Sr))
    
    def waveNumber_opt(self,isPlotting=True):
        KS= self.K[:,2] # Wave Length for S0 n=2
        KA= self.K[:,1] # Wave Length for A0 n=2
        obs=1
        lmbda_a0 = (2*np.pi)/KA
        lmbda_s0 = (2*np.pi)/KS
        X = np.linspace(0.1e-3, 100*5e-3, 10000)
        sigma_Ar= np.zeros_like(X, dtype=complex)
        sigma_Sr= np.zeros_like(X, dtype=complex)

        for i,x in enumerate(X):
            T_A=self.Stress_function_2(KA[obs] ,x)
            T_S=self.Stress_function_2(KS[obs] ,x)
            sigma_Ar[i]=T_A[0]
            sigma_Sr[i]=T_S[0]
        if isPlotting:
            fig, axes=plt.subplots(1,2)
            self.figureplot(X, abs(sigma_Ar), ax=axes[0], ylabel='$\tau_{rz}$', xlabel='WaveLength',label='A0'+str(KA[obs]),title='Wave Number wi',linestyle='-', marker='o', markersize=1, c='b')
            self.figureplot(X, abs(sigma_Sr), ax=axes[1], ylabel='$\tau_{rz}$', xlabel='WaveLength',label='S0'+str(KS[obs]) ,title='Wave Number wi',linestyle='-', marker='o', markersize=1, c='k')

        


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

        return y_pred, model.coef_
    
    def Norm(self,x):
        return x/np.sqrt(np.sum(x**2))

    def inter_fun(self, x, y, xnew):
        f=interp1d(x,y)
        return f(xnew)
    
    def analysis(self):
         Utip=Try.PRHW(isPlotting=True)
         TW=Try.tw_term(isPlotting=False)
         f_new= np.linspace(10e3,1e6 , 1000) 
         Utip=self.inter_fun(self.Freq, Utip, f_new)
         TW=self.inter_fun(self.Freq, TW, f_new)
         plt.figure()
        #  plt.plot(f_new, Utip)
         plt.plot(f_new, TW/Utip)
    def figureplot(self,x,y,ax=None,**keyword):
        '''
        figNumber = is tupple(2,1)
        '''
        A=[]
        XLABEL='F[Hz]'
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
            elif key=='ylim':
                Ylim=keyword[key]
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
                ax.set_title(Title, fontsize=10)
            elif a== 'ylim':
                ax.set_ylim(Ylim)
            elif a =='filename':
                plt.savefig(self.saveFigure+FileName+'.png')

if __name__=='__main__':
    Try=WaveDisplacment()
    # F=Try.Freq
    # Try.waveNumber_opt()
    # Try.optimum_wavenumber()
    # Try.optimum_radiusA()
    # Try.stressComparision_bw_FEM(isPlotting=True)
    # Try.stressFunction_tooptimize()
    # Try.plottingWaveNumber()
    # Try.freePZTimpedance()
    Try.checkRootsTheequation()
    # Utip=Try.PRHW(isPlotting=True)
    # Try.tw_term(isPlotting=True)
    # Try.analysis()
    # Try.Displacement_calulation(isPlotting=True)
    # Try.stressFunction_tooptimize()
    # %%
    # plt.figure()
    # plt.plot(Try.Freq , Utip)
    # Try.tw_term(isPlotting=True)
    # Try.frequency_dependent_stress(isPlotting=True)
    print("---Plot graph finish---")
    plt.show()
    plt.close()

   
    # C= (4*7.346*1e-9*20*1e-3*10*1e-3)/(5*1e-3)
    # F=Try.Freq

# %%
