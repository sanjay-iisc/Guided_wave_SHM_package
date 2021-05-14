import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
"""
Ref[1]== 'Victor Giurgiutiu'book -page no.314
implimantion of wave number calculation for the imaginary and real roots for symmetric and antisymmetric modes.
"""
class  WaveNumber:
    def __init__(self):
        self.d = (1.5/2)*1e-3 # in mm Half thickness 
        self.E = 70e9#68e9 # [kN/mm^2]
        self.nu = 0.33
        self.rho = 2700#[g/cm^3]
        self.tolt=1e-9# tolarance value to get how much precisions you want 
        self.omega_min=0.001
        # self._min=100#100;
        self.incr= 1 #1/100
        self.omega_max=10;
        self.modes=6 ### Number of modes
        ####----Bulk Properties
        self.Lambada = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        self.Mue = self.E / (2 * (1 + self.nu))
        self.C_L = math.sqrt((self.Lambada + 2 * self.Mue) / self.rho)
        self.C_T = math.sqrt(self.Mue / self.rho)
        self.R = self.C_L/self.C_T

    def Eq_symmetricMode(self,k, omega):
        if k >= omega :
            y=self.both_complex_symmetricMode(k,omega)
        elif k < omega and k > ( omega/ self.R):
            y =self.one_Real_and_second_complex_symmetricMode( k, omega)
        else:
            y= self.all_RealsymmetricMode( k, omega)
        return y

    def both_complex_symmetricMode(self, k, omega):
        n_p = np.abs(cmath.sqrt(omega**2/self.R**2 -k**2))
        n_s=np.abs(cmath.sqrt(omega**2 -k**2))
        Ds1= (k**2+n_s**2)**2 * np.cosh(n_p) *np.sinh(n_s) 
        Ds2= -4*(k**2)*n_p*n_s * np.cosh(n_s) *np.sinh(n_p)
        Ds= Ds1+Ds2
        return Ds
    
        
    def one_Real_and_second_complex_symmetricMode(self, k, omega):
        n_p = np.abs(cmath.sqrt(omega**2/self.R**2 -k**2))
        n_s=np.abs(cmath.sqrt(omega**2 -k**2))
        Ds1= (k**2-n_s**2)**2 * np.cosh(n_p) *np.sin(n_s) 
        Ds2= -4*(k**2)*n_p*n_s * np.cos(n_s) *np.sinh(n_p)
        Ds= Ds1+Ds2
        return Ds

    def all_RealsymmetricMode(self, k, omega):
        n_p = np.abs(cmath.sqrt(omega**2/self.R**2 -k**2))
        n_s=np.abs(cmath.sqrt(omega**2 -k**2))
        Ds1= (k**2-n_s**2)**2 * np.cos(n_p) *np.sin(n_s) 
        Ds2= 4*(k**2)*n_p*n_s * np.cos(n_s) *np.sin(n_p)
        Ds= Ds1+Ds2
        return Ds
        
    def SSmode(self,k):
        omega2 = self.omega_min
        incr=  self.incr
        omega0 = np.zeros((self.modes , 2))
        n = 0 
        while omega2 < self.omega_max and n < self.modes :
            omega1 = omega2
            omega2 = omega1 +incr
            y1 = self.Eq_symmetricMode(k,omega1)
            y2 = self.Eq_symmetricMode(k,omega2)
            while y1*y2 > 0 and omega2 <self.omega_max:
                omega1 =omega2
                omega2 = omega1 +incr
                y1 = self.Eq_symmetricMode(k,omega1)
                y2 = self.Eq_symmetricMode(k,omega2)
                ttt=y1
            if y1*y2 < 0 :
                # print('sss')
                omega0[n,:]=np.array([omega1,omega2])
                n= n+1
            else:
                if y1 ==0 and np.isclose(y2,0):
                    
                    omega0[n,:]=np.array([omega1,omega1])
                    n = n+1
                
                elif y2 ==0 and np.isclose(y1,0):
                    
                    omega0[n,:]=np.array([omega2,omega2])
                    n=n+1
                
                elif y1==0 and y2==0:
                    
                    omega0[n,:]=np.array([omega1,omega1])
                    n=n+1
                    
                    omega0[n,:]=np.array([omega2,omega2])
                    n=n+1
                    omega2=omega2+1
        
        return omega0 , n
    
    def Sym_rootCalculation(self,cp1,cp2,fd,N=100000):
        while (cp2-cp1)> self.tolt:
            y1=self.Eq_symmetricMode(fd,cp1)
            y2 =self.Eq_symmetricMode(fd,cp2)
            cp0 = (cp1+cp2)/2
            y0 =self.Eq_symmetricMode(fd,cp0)
            if y1*y0 < 0 :
                cp2=cp0
            elif y2*y0 <0:
                cp1=cp0
            elif y0==0:
                break
            elif y1==0:
                cp2=cp1
            elif y2 ==0:
                cp1=cp2
        return (cp1+cp2)/2
            
    
    def SymmetricMode(self):
         K =np.linspace(0.025,5,500)
         omegaS=np.zeros((len(K),self.modes));
         for i, k in enumerate(K):
             # first loop on the frequency
             print('The loop =',i)
             omega12 , n = self.SSmode(k)
             for j in range(n):
                 cp1=omega12[j,0]
                 cp2=omega12[j,1]
                 omegaS[i,j]=self.Sym_rootCalculation(cp1,cp2,k)
                 print('loop=',j) 
         return omegaS, K
    
    ##################Anti symmetric--Real Roots
 
    def Eq_AntisymmetricMode(self,k, omega):
        if k >= omega :
            y=self.both_complex_AntisymmetricMode(k,omega)
        elif k < omega and k > ( omega/ self.R):
            y =self.one_Real_and_second_complex_AntisymmetricMode( k, omega)
        else:
            y= self.all_RealAntisymmetricMode( k, omega)
        return y        
        
    def both_complex_AntisymmetricMode(self, k, omega):
        n_p = np.abs(cmath.sqrt(omega**2/self.R**2 -k**2))
        n_s=np.abs(cmath.sqrt(omega**2 -k**2))
        Da1= (k**2+n_s**2)**2 * np.sinh(n_p) *np.cosh(n_s) 
        Da2= -4*(k**2)*n_p*n_s * np.sinh(n_s) *np.cosh(n_p)
        Da= Da1+Da2
        return Da
        
    def one_Real_and_second_complex_AntisymmetricMode(self, k, omega):
        n_p = np.abs(cmath.sqrt(omega**2/self.R**2 -k**2))
        n_s=np.abs(cmath.sqrt(omega**2 -k**2))
        Da1= (k**2-n_s**2)**2 * np.sinh(n_p) *np.cos(n_s) 
        Da2= 4*(k**2)*n_p*n_s * np.sin(n_s) *np.cosh(n_p)
        Da= Da1+Da2
        return Da
    

    def all_RealAntisymmetricMode(self, k, omega):
        n_p = np.abs(cmath.sqrt(omega**2/self.R**2 -k**2))
        n_s=np.abs(cmath.sqrt(omega**2 -k**2))
        Da1= (k**2-n_s**2)**2 * np.sin(n_p) *np.cos(n_s) 
        Da2= 4*(k**2)*n_p*n_s * np.sin(n_s) *np.cos(n_p)
        Da= Da1+Da2
        return Da

    def AAmode(self,k):
        omega2 = self.omega_min
        incr=  self.incr
        omega0 = np.zeros((self.modes , 2))
        n = 0 
        while omega2 < self.omega_max and n < self.modes :
            omega1 = omega2
            omega2 = omega1 +incr
            y1 = self.Eq_AntisymmetricMode(k,omega1)
            y2 = self.Eq_AntisymmetricMode(k,omega2)
            
            while y1*y2 > 0 and omega2 <self.omega_max:
                omega1 =omega2
                omega2 = omega1 +incr
                y1 = self.Eq_AntisymmetricMode(k,omega1)
                y2 = self.Eq_AntisymmetricMode(k,omega2)
                ttt=y1
            
            if y1*y2 < 0 :
                # print('sss')
                omega0[n,:]=np.array([omega1,omega2])
                n= n+1
            else:
                if y1 ==0 and np.isclose(y2,0):
                    
                    omega0[n,:]=np.array([omega1,omega1])
                    n = n+1
                
                elif y2 ==0 and np.isclose(y1,0):
                    
                    omega0[n,:]=np.array([omega2,omega2])
                    n=n+1
                
                elif y1==0 and y2==0:
                    
                    omega0[n,:]=np.array([omega1,omega1])
                    n=n+1
                    
                    omega0[n,:]=np.array([omega2,omega2])
                    n=n+1
                    omega2=omega2+1
        
        return omega0 , n
        
    
    
    
    def AntiSym_rootCalculation(self,cp1,cp2,fd,N=100000):
        while (cp2-cp1)> self.tolt:
            y1=self.Eq_AntisymmetricMode(fd,cp1)
            y2 =self.Eq_AntisymmetricMode(fd,cp2)
            # print(sanj)
            cp0 = (cp1+cp2)/2
            y0 =self.Eq_AntisymmetricMode(fd,cp0)
            if y1*y0 < 0 :
                cp2=cp0
            elif y2*y0 <0:
                cp1=cp0
            elif y0==0:
                break
            elif y1==0:
                cp2=cp1
            elif y2 ==0:
                cp1=cp2
        return (cp1+cp2)/2
                
    
    
    def AntiSymmetricMode(self):
         KA =np.linspace(0.3125,5,1000)
         omegaA=np.zeros((len(KA),self.modes));
         for i, k in enumerate(KA):
             # first loop on the frequency
             print('The loop =',i)
             omega12 , n = self.AAmode(k)
             for j in range(n):
                 cp1=omega12[j,0]
                 cp2=omega12[j,1]
                 omegaA[i,j]=self.AntiSym_rootCalculation(cp1,cp2,k)
                 print('loop=',j) 
         return omegaA, KA
    #-------Evanscant modes --for pure imaginary-----------------
    ## Evanscant--symmetric
    def EvenScent_Eq_symmetricMode(self,k, omega):
        y= self.EvenScent_all_RealsymmetricMode( k, omega)
        return y
    def EvenScent_all_RealsymmetricMode(self, k, omega):
        n_p = np.abs(cmath.sqrt( (omega**2/self.R**2) +k**2))
        n_s=np.abs(cmath.sqrt(omega**2 +k**2))
        Ds1= (-k**2-n_s**2)**2 * np.cos(n_p) *np.sin(n_s) 
        Ds2= -4*(k**2)*n_p*n_s * np.cos(n_s) *np.sin(n_p)
        Ds= Ds1+Ds2
        return Ds
    def EvenScent_SSmode(self,k):
        omega2 = self.omega_min
        incr=  self.incr
        omega0 = np.zeros((self.modes , 2))
        n = 0 
        
        while omega2 < self.omega_max and n < self.modes :
            omega1 = omega2
            omega2 = omega1 +incr
            y1 = self.EvenScent_Eq_symmetricMode(k,omega1)
            y2 = self.EvenScent_Eq_symmetricMode(k,omega2)
            
            while y1*y2 > 0 and omega2 <self.omega_max:
                omega1 =omega2
                omega2 = omega1 +incr
                y1 = self.EvenScent_Eq_symmetricMode(k,omega1)
                y2 = self.EvenScent_Eq_symmetricMode(k,omega2)
                ttt=y1
            
            if y1*y2 < 0 :
                # print('sss')
                omega0[n,:]=np.array([omega1,omega2])
                n= n+1
            else:
                if y1 ==0 and np.isclose(y2,0):
                    
                    omega0[n,:]=np.array([omega1,omega1])
                    n = n+1
                
                elif y2 ==0 and np.isclose(y1,0):
                    
                    omega0[n,:]=np.array([omega2,omega2])
                    n=n+1
                
                elif y1==0 and y2==0:
                    
                    omega0[n,:]=np.array([omega1,omega1])
                    n=n+1
                    
                    omega0[n,:]=np.array([omega2,omega2])
                    n=n+1
                    omega2=omega2+1
        
        return omega0 , n
    
    def EvenScent_Sym_rootCalculation(self,cp1,cp2,fd,N=100000):
        while (cp2-cp1)> self.tolt:
            y1=self.EvenScent_Eq_symmetricMode(fd,cp1)
            y2 =self.EvenScent_Eq_symmetricMode(fd,cp2)
            cp0 = (cp1+cp2)/2
            y0 =self.EvenScent_Eq_symmetricMode(fd,cp0)
            if y1*y0 < 0 :
                cp2=cp0
            elif y2*y0 <0:
                cp1=cp0
            elif y0==0:
                break
            elif y1==0:
                cp2=cp1
            elif y2 ==0:
                cp1=cp2
        return (cp1+cp2)/2
    
    def EvenScent_SymmetricMode(self):
         K =np.linspace(0.2,10,1000) # This assumed as the -ij*k or +ij*k
         omegaS=np.zeros((len(K),self.modes));
         SortingModes=[]
         for i, k in enumerate(K):
             print('The Loop Positions=',i)
             omega12 , n = self.EvenScent_SSmode(k)
             for j in range(n):
                 cp1=omega12[j,0]
                 cp2=omega12[j,1]
                 omegaS[i,j]=self.EvenScent_Sym_rootCalculation(cp1,cp2,k)
                 print('loop=',j) 
                 SortingModes.append((k,omegaS[i,j]))
         return omegaS, K
    ## Evanscant--Antisymmetric
    def EvenScent_AntiSymmetricMode(self):
         K =np.linspace(0.01,5,1000) # This assumed as the -ij*k or +ij*k
         omegaS=np.zeros((len(K),self.modes));
         SortingModes=[]=[]
         for i, k in enumerate(K):
             # first loop on the frequency
             print('The Loop Positions=',i)
             omega12 , n = self.AntiEvenScent_AAmode(k)
             for j in range(n):
                 cp1=omega12[j,0]
                 cp2=omega12[j,1]
                 omegaS[i,j]=self.EvenScent_AntiSym_rootCalculation(cp1,cp2,k)                        
                 SortingModes.append((k,omegaS[i,j]))
                 print('loop=',j) 
         return omegaS, K
    
    def EvenScent_Eq_AntisymmetricMode(self,k, omega):
        y= self.EvenScent_all_RealAntisymmetricMode( k, omega)
        return y

    def EvenScent_all_RealAntisymmetricMode(self, k, omega):
        n_p = np.abs(cmath.sqrt( (omega**2/self.R**2) +k**2))
        n_s=np.abs(cmath.sqrt(omega**2 +k**2))
        Da1= (-k**2-n_s**2)**2 * np.sin(n_p) *np.cos(n_s) 
        Da2= -4*(k**2)*n_p*n_s * np.sin(n_s) *np.cos(n_p)
        Da= Da1+Da2
        return Da
    
    
        
    def AntiEvenScent_AAmode(self,k):
        omega2 = self.omega_min
        incr=  self.incr
        omega0 = np.zeros((self.modes , 2))
        n = 0 
        
        while omega2 < self.omega_max and n < self.modes :
            omega1 = omega2
            omega2 = omega1 +incr
            y1 = self.EvenScent_Eq_AntisymmetricMode(k,omega1)
            y2 = self.EvenScent_Eq_AntisymmetricMode(k,omega2)
            
            while y1*y2 > 0 and omega2 <self.omega_max:
                omega1 =omega2
                omega2 = omega1 +incr
                y1 = self.EvenScent_Eq_AntisymmetricMode(k,omega1)
                y2 = self.EvenScent_Eq_AntisymmetricMode(k,omega2)
                ttt=y1
            
            if y1*y2 < 0 :
                # print('sss')
                omega0[n,:]=np.array([omega1,omega2])
                n= n+1
            else:
                if y1 ==0 and np.isclose(y2,0):
                    
                    omega0[n,:]=np.array([omega1,omega1])
                    n = n+1
                
                elif y2 ==0 and np.isclose(y1,0):
                    
                    omega0[n,:]=np.array([omega2,omega2])
                    n=n+1
                
                elif y1==0 and y2==0:
                    
                    omega0[n,:]=np.array([omega1,omega1])
                    n=n+1
                    
                    omega0[n,:]=np.array([omega2,omega2])
                    n=n+1
                    omega2=omega2+1
        
        return omega0 , n
        
    
    def EvenScent_AntiSym_rootCalculation(self,cp1,cp2,fd,N=100000):
        while (cp2-cp1)> self.tolt:
            y1=self.EvenScent_Eq_AntisymmetricMode(fd,cp1)
            y2 =self.EvenScent_Eq_AntisymmetricMode(fd,cp2)
            cp0 = (cp1+cp2)/2
            y0 =self.EvenScent_Eq_AntisymmetricMode(fd,cp0)
            if y1*y0 < 0 :
                cp2=cp0
            elif y2*y0 <0:
                cp1=cp0
            elif y0==0:
                break
            elif y1==0:
                cp2=cp1
            elif y2 ==0:
                cp1=cp2
        
        return (cp1+cp2)/2
    
    def coversion_omega(self,x):
        return (x*self.C_T)/self.d
    def coversion_waveNumber(self,x):
        return (x)/self.d
        
    ## Computation of the WaveNumber 
    def both_modes_Sym_and_Anti(self, saveData=False):
        print("-----------------------------Start the Anti-Symmetric Modes------------------------")
        omegaA, KA=self.AntiSymmetricMode()
        print("-----------------------------Start the Symmetric Modes-----------------------------")
        omegaS, KS=self.SymmetricMode()
        print("-----------------------------Start the Evan-Anti-Symmetric Modes-----------------------------")
        # evan_omegaA,evan_KA=self.EvenScent_AntiSymmetricMode()
        print("-----------------------------Start the Evan-Symmetric Modes-----------------------------")
        # evan_omegaS,evan_KS=self.EvenScent_SymmetricMode()
        print("-----------------------------Process Completed-----------------------------")
    ef ds_equation(self,K,W):
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

    def plotting_modes(self,x,y):
        plt.figure()
        for i in range(self.modes):
            plt.plot(x, y[:,i], linestyle='None', marker='o', label='mode: '+str(i),markersize=1)
        plt.ylabel('F[KHz]')
        plt.xlabel('WaveNumber')
        plt.title('symmetric-waveNumber')
        plt.legend()

if __name__=='__main__':
    Symmetric_Wavenumber=WaveNumber()
    omegaS, KS=self.SymmetricMode()
    
    # omegaS,K=Symmetric_Wavenumber.EvenScent_AntiSymmetricMode()
    # Symmetric_Wavenumber.both_modes_Sym_and_Anti()
    # Symmetric_Wavenumber.plotting_modes(K,omegaS)
    plt.show()
