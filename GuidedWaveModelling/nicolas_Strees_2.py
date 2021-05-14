import numpy as np
import Hybridmodel as HM
import Figure_plot as graph
import matplotlib.pyplot as plt
import scipy.special 
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

ttt_ar=np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAR.npy")
ttt_az=np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressAZ.npy")
ttt_sr=np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSR.npy")
ttt_sz=np.load("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\stressSZ.npy")



