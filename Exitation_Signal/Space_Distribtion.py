import sys
sys.path.append("./")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special  import jv
import GuidedWaveModelling.Figure_plot as graph
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from hankel import HankelTransform   
import hankel
#---------------------------------------------------------------
plt.close()
#-------------Import the optimized stress profile ----------------------------------
path_stress_optimised= "K:\LMC\Sanjay\Code\Optimization\optimization_stress\Optimized_stress_const_RR\\stress_hyperperameter_RR.csv"
Stress_opt=pd.read_csv(path_stress_optimised)
Stress_opt_mean=Stress_opt.mean(axis=0)

#------------------------------------------------function defined in the optimization----------------------------
a=5e-3
K=np.linspace(0,3000,10000)
# r=np.linspace(0.1e-3,30e-3,1000)
# fr= lambda K : np.where( K < 1e7, jv(Stress_opt_mean['alpha'],K*Stress_opt_mean['gamma']*a)*(K*a)**Stress_opt_mean['beta'],0)
# fig,axes=plt.subplots(1,1)
# graph.figureplot(K, fr(K),ax=axes,title=' Optimized Stress Profile-Tr ')
# print(Stress_opt_mean)
# #--------------------------- Inverse Hankel Transformation of the fr -----------------------------------------------
# h    = HankelTransform(nu=1,N=8000,h=5e-8)
# tx = h.transform(fr, r, False, inverse=True)
# best_h, result, best_N = hankel.get_h(
#     fr,
#     nu=1
# )
# print(f"best_h = {best_h}, best_N={best_N}")
# # print("Relative Error: ", result/res - 1)

# fig,axes=plt.subplots(1,1)
# graph.figureplot(r, tx,ax=axes,title=' Optimized Stress Profile-Space ')
plt.show()