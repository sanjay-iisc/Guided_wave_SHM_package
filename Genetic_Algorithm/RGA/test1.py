import main as GA
import numpy as np
import matplotlib.pyplot as plt
def fz (x):
        # p1,p2,p3=x
        return np.sum(np.array(x)**2)
GA.GeneticAlgorithm_Base._get_userInputs(fz,dim=3,max_intr=100,population_size=100)
# print('sa'
GA1=GA.GA_strat()
GA1.RUN()
plt.show(block=False)
