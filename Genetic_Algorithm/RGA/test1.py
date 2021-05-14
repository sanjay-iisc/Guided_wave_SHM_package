import main as GA
import numpy as np
def fz (x):
        # p1,p2,p3=x
        return np.sum(-np.array(x)**2)
GA.GeneticAlgorithm_Base._get_userInputs(fz,dim=1,max_intr=50,population_size=8)
# print('sa'
GA.GA_RUN()
