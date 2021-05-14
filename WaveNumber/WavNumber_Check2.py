# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# plt.style.use('.\Plotting_style\science.mplstyle')
#%%
class Rootscheck:
    def __init__(self):
        self.d = (1.5/2)*1e-3 # in mm Half thickness 
        self.E = 70e9#68e9 # [kN/mm^2]
        self.nu = 0.33
        self.rho = 2700#[g/cm^3]
        ####----Bulk Properties
        self.Lambada = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        self.Mue = self.E / (2 * (1 + self.nu))
        self.C_L = np.sqrt((self.Lambada + 2 * self.Mue) / self.rho)
        self.C_T = np.sqrt(self.Mue / self.rho)
        self.R = self.C_L/self.C_T
        self.saveFigure="E:\PPT\Presentation\\04052021_ppt\Figure\\"
    def Ds_equation(self, k, omega):
        n_p = np.lib.scimath.sqrt(omega**2/self.R**2 -k**2) 
        n_s=np.lib.scimath.sqrt(omega**2 -k**2)
        Ds1= (k**2-n_s**2)**2 * np.cos(n_p) *np.sin(n_s) 
        Ds2= 4*(k**2)*n_p*n_s * np.cos(n_s) *np.sin(n_p)
        Ds= Ds1+Ds2
        return Ds

    def Da_equation(self, k, omega):
        n_p = np.lib.scimath.sqrt(omega**2/self.R**2 -k**2)
        n_s=np.lib.scimath.sqrt(omega**2 -k**2)
        Da1= (k**2-n_s**2)**2 * np.sin(n_p) *np.cos(n_s) 
        Da2= 4*(k**2)*n_p*n_s * np.sin(n_s) *np.cos(n_p)
        Da= Da1+Da2
        return Da
   
    def ds_roots_check_convergence(self):
        K1= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix.npy")*1e3)#rad/mm
        Freq1 = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix.npy")*1e6#MHz
        K2= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix_10mesh.npy")*1e3)#rad/mm
        Freq2 = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix_10mesh.npy")*1e6#MHz
        K3= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix_1mesh.npy")*1e3)#rad/mm
        Freq3 = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix_1mesh.npy")*1e6#MHz
        K4=np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrixbastien.npy") 
        #(np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix_10mesh.npy")*1e3)#rad/mm
        Freq4 =np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrixbastien.npy")[:,0] #np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix_10mesh.npy")*1e6#MHz
        K_data=[K1,K4]#K2,K3,,K2]
        Freq_data=[Freq1,Freq4]#Freq2,Freq3,#,Freq2]
        Alpha=[4, 2, 1,1]
        Name=['Jorden(d/100)','Bastien'] #'Jorden(d/10)','Jorden(d/1)'
        Index=0
        f=0.01e6
        fig, axes=plt.subplots()
        for K,Freq in zip(K_data,Freq_data):
            ffs = np.argmin(np.abs(f-Freq))
            for i in [ffs]:#np.arange(100,900,200):
                omega =2*np.pi*Freq
                K_norm=K*self.d
                omega_norm=(omega*self.d)/self.C_T
                Ds=self.Ds_equation(K_norm[i,:],omega_norm[i])
                self.figureplot(np.arange(len(K_norm[i,:])),abs(Ds),ax=axes, ylabel=r'$\bar{Ds}$', xlabel='#modes',
                label='Freq :'+str(np.round(Freq[i]/1e6,2))+' MHz-'+Name[Index], marker='o', linestyle='None', markersize=Alpha[Index]
                , filename='Dsfreq='+str(f))
            Index+=1
    def da_roots_check_convergence(self):
        K1= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix.npy")*1e3)#rad/mm
        Freq1 = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix.npy")*1e6#MHz
        K2= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix_10mesh.npy")*1e3)#rad/mm
        Freq2 = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix_10mesh.npy")*1e6#MHz
        K3= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix_1mesh.npy")*1e3)#rad/mm
        Freq3 = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix_1mesh.npy")*1e6#MHz
        K4=np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrixbastien.npy") 
        #(np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix_10mesh.npy")*1e3)#rad/mm
        Freq4 =np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrixbastien.npy")[:,0] #np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix_10mesh.npy")*1e6#MHz
        K_data=[K1,K4]#K2,K3,,K2]
        Freq_data=[Freq1,Freq4]#Freq2,Freq3,#,Freq2]
        Alpha=[4, 2, 1,1]
        Name=['Jorden(d/100)','Bastien'] #'Jorden(d/10)','Jorden(d/1)'
        Index=0
        f=0.01e6
        fig, axes=plt.subplots()
        for K,Freq in zip(K_data,Freq_data):
            ffs = np.argmin(np.abs(f-Freq))
            for i in [ffs]:#np.arange(100,900,200):
                omega =2*np.pi*Freq
                K_norm=K*self.d
                omega_norm=(omega*self.d)/self.C_T
                Da=self.Da_equation(K_norm[i,:],omega_norm[i])
                self.figureplot(np.arange(len(K_norm[i,:])),abs(Da),ax=axes, ylabel=r'$\bar{Da}$', xlabel='#modes',
                label='Freq :'+str(np.round(Freq[i]/1e6,2))+' MHz-'+Name[Index], marker='o', linestyle='None', markersize=Alpha[Index]
                , filename='Dafreq='+str(f))
            Index+=1
    def ds_roots_check_convergence_modes(self):
        K1= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix.npy")*1e3)#rad/mm
        Freq1 = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix.npy")*1e6#MHz
        
        K2= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix_10mesh.npy")*1e3)#rad/mm
        Freq2 = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix_10mesh.npy")*1e6#MHz
        K3= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix_1mesh.npy")*1e3)#rad/mm
        Freq3 = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix_1mesh.npy")*1e6#MHz
        K_data=[K1,K2,K3]#,K2]
        Freq_data=[Freq1,Freq2,Freq3]#,Freq2]
        D=[100,10,1]
        Alpha=[4, 2,1]
        
        
        for i in [0,1,2,3,4,23,24,25,26]:
            fig, axes=plt.subplots()
            Index=0
            for K,Freq in zip(K_data,Freq_data):
                omega =2*np.pi*Freq
                K_norm=K*self.d
                omega_norm=(omega*self.d)/self.C_T
                Ds=self.Ds_equation(K_norm[:,i],omega_norm)
                self.figureplot(Freq,abs(Ds),ax=axes, ylabel=r'$\bar{Ds}$', 
                label='-Meshsize :'+'d/'+str(D[Index]), marker='o', linestyle='None', alpha=1,markersize=Alpha[Index],
                title='Mode:'+str(i), filename='Ds'+'Mode'+str(i))
                Index+=1

    def da_roots_check_convergence_modes(self):
        K1= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix.npy")*1e3)#rad/mm
        Freq1 = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix.npy")*1e6#MHz
        
        K2= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix_10mesh.npy")*1e3)#rad/mm
        Freq2 = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix_10mesh.npy")*1e6#MHz
        K3= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix_1mesh.npy")*1e3)#rad/mm
        Freq3 = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix_1mesh.npy")*1e6#MHz
        K_data=[K1,K2,K3]#,K2]
        Freq_data=[Freq1,Freq2,Freq3]#,Freq2]
        D=[100,10,1]
        Alpha=[4,2, 1]
        
        
        
        for i in [0,1,2,3,4,23,24,25,26]:
            fig, axes=plt.subplots()
            Index=0
            for K,Freq in zip(K_data,Freq_data):
            #np.arange(0,,):#np.arange(100,900,200):
                omega =2*np.pi*Freq
                K_norm=K*self.d
                omega_norm=(omega*self.d)/self.C_T
                Da=self.Da_equation(K_norm[:,i],omega_norm)
                self.figureplot(Freq,abs(Da),ax=axes, ylabel=r'$\bar{Da}$', 
                label='-Meshsize :'+'d/'+str(D[Index]), marker='o', linestyle='None', alpha=1,markersize=Alpha[Index],
                title='Mode:'+str(i), filename='Da'+'Mode'+str(i))
           
                Index+=1
                

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
    Model=Rootscheck()
    Model.ds_roots_check_convergence()
    Model.da_roots_check_convergence()
    # Model.da_roots_check_convergence_modes()
    #one h/10
    # Model2=Rootscheck()
    
    # Model2.ds_roots_check()
    # Ds=Model.Ds_equation(Model.K_norm,Model.omega_norm)
    # fig, axes=plt.subplots()
    # Model.figureplot(Model.Freq,abs(Ds),ax=axes)
    plt.show()






# # %%
# path ='E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate'
# filePathpro = os.path.join(path, 'modes_15mmFinner_PRO.csv')
# filePathevn = os.path.join(path, 'modes_15mm_EVN.csv')
# waveNumber_pro = pd.read_csv(filePathpro)
# waveNumber_evn = pd.read_csv(filePathevn)
# freqDisper= np.array(waveNumber_pro['Frequency(MHz)']) #in khz
# jordanka =  np.array(waveNumber_pro['A0'])*1e3 #in rad/m
# jordanks =  np.array(waveNumber_pro['S0'])*1e3
# jordanksh = np.array(waveNumber_pro['SH'])*1e3
# evenJordan=np.array(waveNumber_evn['Num=0'])*1e3
# ## Analytical WaveNumber 
# #%%PropagatingModes
# Ana_filePathpro_S0 = os.path.join(path, 'Analytical_15mmplate_S_mode.csv')
# Ana_filePathpro_A0 = os.path.join(path, 'Analytical_15mmplate_A_mode.csv')
# waveNumber_S = pd.read_csv(Ana_filePathpro_S0)
# waveNumber_A = pd.read_csv(Ana_filePathpro_A0)
# Ana_S0 = waveNumber_S ['S0[rad/m]']
# Ana_A0 = waveNumber_A ['A0[rad/m]']
# Freq= waveNumber_A ['F[hz]']*1e-6
# #%%Evenscant Modes
# Ana_filePathpro_evenA0 = os.path.join(path, 'Analytical_15mmplate_Even_A_mode.csv')
# EvenwaveNumber_A = pd.read_csv(Ana_filePathpro_evenA0)
# evenAna_A0=EvenwaveNumber_A ['A[rad/m]']

# fig,ax=plt.subplots(1,1, figsize=(6,5))
# fig.axes[0].scatter(Freq,Ana_S0,s=0.1, label = 'Ana-S0',color='k', marker='o', alpha=0.5)
# fig.axes[0].scatter(Freq,Ana_A0,s=0.1, label = 'Ana-A0',color='k', marker='o', alpha=0.5)
# fig.axes[0].scatter(Freq,-evenAna_A0,s=0.5, label = 'Imag-A0',color='k', marker='o', alpha=0.5)

# fig.axes[0].plot(freqDisper,jordanks, label='SFEM-S0', alpha=0.5, linewidth=3)
# fig.axes[0].plot(freqDisper,jordanka, label='SFEM-A0', alpha=0.5,linewidth=3)   
# fig.axes[0].scatter(freqDisper,evenJordan, label='imag-SFEM-A0', alpha=0.5,s=0.2)   

# fig.axes[0].legend()
# fig.axes[0].set_ylabel('Wave Number [rad/M]')
# # # plt.show()
# # K= (np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\WaveNumberMatrix.npy")*1e3)[:-100, :]#rad/mm
        
# # Freq = np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\Freq_WaveNumberMatrix.npy")[:-100]*1e6#MHz
        
# def plotiing(x,y, Label ='Da'):
#     plt.figure()
#     plt.plot(x,y, label=Label)
   
#     plt.legend()
    
# #%%
# def equation_check_jorden_code():
#     Ds_jor, Da_jor=[],[]
    
#     for i,freq__ in enumerate(Freq) :
#         ds,da=dads_equation(Ana_A0,freq__)
#         # ds,da=dads_equation(Ana_A0,freq__)
#         Ds_jor.append(ds)
#         Da_jor.append(da)
#     return Ds_jor ,Da_jor
# Ds_jor, Da_jor=equation_check_jorden_code()

# plotiing(Freq ,abs(Da_jor[100]), Label='Da')
# plotiing(Freq ,abs(Ds_jor[100]), Label='Ds')
# plt.show()
