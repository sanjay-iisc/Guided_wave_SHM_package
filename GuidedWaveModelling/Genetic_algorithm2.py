import Hybridmodel as HM
import Figure_plot as graph
import matplotlib.pyplot as plt
import scipy.special 
from scipy.special  import jv
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import count
plt.style.use('fivethirtyeight')
POPULATION_SIZE=1000
TOURNAMENT_SELECTION_SIZE=2
MUTATION_RATE=0.25
NUMBER_OF_ELITE_CHROMOSOMES=0
sizeList=[10,10,10]
c
a=0.5 # alpha for crossover
nc=2
sigma=[0.5,0.1,0.1,0.1]


f=100e3
# Importing the wave stress
Freq = np.arange(5, 1000, 5)*1e3 # Hz
p= np.argmin(abs(Freq-f))
Rz_waveNumber=pd.read_csv("E:\Work\Code\matlabJordan\calcul_modal\\NicolasPlate\FEMstress\data_stress_RZ_waveNumber.csv")
stress_KRz=Rz_waveNumber['sigma_RZ[N/mm^2] '+'F='+str(int(Freq[p-1]*1e-3))+' [KHz]']
K_rz=Rz_waveNumber['K[rad/mm]']*1e3
K=np.linspace(10,3000,100)
stress_KRz=interp1d(K_rz, stress_KRz)(K)

#############-----
Model=HM.t_w()
Ks=Model._tipDisp._equations.K[:,2]
Ka=Model._tipDisp._equations.K[:,1]
x=Model._tipDisp._equations.Freq
a=Model._tipDisp._equations.a
# Admittance term
Aw=Model.constan_term(isPlotting=False)
fAw=interp1d(x,Aw)
# t_r=x1*a*jv(1,K*a*x3)+x2*fAw(f)*a*jv(2,K*a*x3)/K
def demo_func(p):
    x1,x2,x3=p
    t_r=x1*a*jv(1,K*a*x3)+x2*fAw(f)*a*jv(2,K*a*x3)/K
    return np.square((stress_KRz-t_r)).sum()/np.square((stress_KRz)).sum()

# x_true = np.linspace(-1.2, 1.2, 30)
# y_true = x_true ** 3 - x_true + 0.4 * np.random.rand(30)

# def f_fun(x, a, b, c, d):
#     return a * x ** 3 + b * x ** 2 + c * x + d
# def demo_func(p):
#     a, b, c, d = p
#     residuals = np.square(f_fun(x_true, a, b, c, d) - y_true).sum()
#     return residuals
# Minmization -- at the function : [-0.0003226546544196048, 0.05041568080767159, 0.5000068037060492]
# [-2.98023233e-08  4.99999898e-02  1.00000000e+00]

class Chromosome:
    def __init__(self):
        self._genes=[]
        self._fitness=0
        for i in range(xUpperList.__len__()):
            a=np.random.uniform(low=xLowerList[i], high=xUpperList[i])
            self._genes.append(a)

    def get_fitness(self):
        self._fitness=demo_func(self._genes)#100*(self._genes[1]-self._genes[0]**2)**2+(1-self._genes[0])**2 
        return self._fitness

    def get_genes(self):
        return self._genes
    
    def __str__(self):
        return self._genes.__str__()

class Population:
    def __init__(self,size):
        self._chromosomes=[]
        i=0
        while i< size:
            self._chromosomes.append(Chromosome())
            i+=1
    def get_chromosomes(self):return self._chromosomes

class GeneticAlgorithm:
    @staticmethod
    def evolve(pop):
        # return GeneticAlgorithm._crossover_population(pop)
        newPop=GeneticAlgorithm._mutate_population(GeneticAlgorithm._crossover_population(pop))
        return GeneticAlgorithm._survivorStage(pop,newPop)
    
    #creating the Crossover Population:
    @staticmethod
    def _crossover_population(pop):
        crossover_pop=Population(0) # HERE I DEFINED the empty chromosomes when put the zero
        for i in range(NUMBER_OF_ELITE_CHROMOSOMES):
            crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i]) ## we will move this chromose to the next gen
        i=NUMBER_OF_ELITE_CHROMOSOMES
        ## here we will exclude the elite chromsome
        while i < POPULATION_SIZE:
            # crossover population will have population after the selection.Then we select the best 2 Chromosome
            chromosome1=GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0] # here we call to the _tournament_population
            chromosome2=GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0]
            crossover_pop.get_chromosomes().append(GeneticAlgorithm._blx_alpha_crossover(chromosome1, chromosome2))
            i+=1
        return crossover_pop
    #creating the Mutation Population:
    @staticmethod
    def _mutate_population(pop):
        for i in range(NUMBER_OF_ELITE_CHROMOSOMES,POPULATION_SIZE):# EXCLUDE THE ELITE CHROMOSOME
            GeneticAlgorithm._Normaldistribution_mutate_chromosome(pop.get_chromosomes()[i])
        return pop
    
    @staticmethod
    def sbx_crossover_chromosomes(chromosome1,chromosome2): # this method does the random gen selection from each one of the parent 
        ## chromosomes
        crossover_chrom = Chromosome()
        # Loop over the Number-variable
        for i in range(chromosome1.get_genes().__len__()):
            ### have to select the two parent and assumption is the p1 < p2 always
            x1=chromosome1.get_genes()[i] 
            x2=chromosome2.get_genes()[i]
            if x2 > x1:
                p2=x2
                p1=x1
            else:
                p2=x1
                p1=x2
            # dt= a*(p2-p1) # define dt 
            u=random.random() # randomly picked the 
            # calculation of beta 
            if u<=0.5:
                beta=(2*u)**(1 / (nc+1))
            else:
                beta=(1/ (2* (1-u)) )**(1 / (nc+1))
            o1 = 0.5* (  (p1+p2)-beta*(p2-p1) )
            o2 = 0.5* (  (p1+p2)+beta*(p2-p1) )
            crossover_chrom.get_genes()[i]=p1*(1-gamma)+gamma*p2
            # print(crossover_chrom.get_genes()[i])
        return crossover_chrom
            

    
    @staticmethod
    def _blx_alpha_crossover(chromosome1,chromosome2):
        crossover_chrom = Chromosome()
        # Loop over the Number-variable
        for i in range(chromosome1.get_genes().__len__()):
            ### have to select the two parent and assumption is the p1 < p2 always
            x1=chromosome1.get_genes()[i] 
            x2=chromosome2.get_genes()[i]
            if x2 > x1:
                p2=x2
                p1=x1
            else:
                p2=x1
                p1=x2
            # dt= a*(p2-p1) # define dt 
            u=random.random() # randomly picked the 
            gamma=(1+2*a)*u-a
            crossover_chrom.get_genes()[i]=p1*(1-gamma)+gamma*p2
            # print(crossover_chrom.get_genes()[i])
        return crossover_chrom
            

    @staticmethod
    def _Normaldistribution_mutate_chromosome(chromosome):
        for i in range(chromosome.get_genes().__len__()):
            chromosome.get_genes()[i]=chromosome.get_genes()[i]+np.random.normal(0, sigma[i])
            if chromosome.get_genes()[i] > xUpperList[i]:
                chromosome.get_genes()[i]=xUpperList[i]
            if chromosome.get_genes()[i] < xLowerList[i]:
                chromosome.get_genes()[i]=xLowerList[i]


    @staticmethod
    def _select_tournament_population(pop):
        tournament_pop=Population(0)
        i=0
        while i < TOURNAMENT_SELECTION_SIZE:
            tournament_pop.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0,POPULATION_SIZE)])
            i+=1
        tournament_pop.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
        return tournament_pop
    @staticmethod
    
    def _survivorStage(oldpop,newpop):
        survivalPop=Population(0)
        oldpop.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
        newpop.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
        n=POPULATION_SIZE//2
        i=0
        while i < n:
            survivalPop.get_chromosomes().append(oldpop.get_chromosomes()[i])
            i+=1
        j=0
        while j < n:
            survivalPop.get_chromosomes().append(newpop.get_chromosomes()[j])
            j+=1
        return survivalPop



def _print_population(pop, gen_number):
    print('\n------------------------------------------------')
    print("Generation #", gen_number, "|Fittest Chromosome fitness :", pop.get_chromosomes()[0].get_fitness())
    print("---------------------------------------------------")
    i=0
    for x in pop.get_chromosomes():
        print("Chromosome #",i,":", x, "| fitness :", x.get_fitness())
        i+=1
population=Population(POPULATION_SIZE)
population.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
_print_population(population,0)
# newpopulation=GeneticAlgorithm().evolve(population)
# newpopulation.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
# _print_population(newpopulation,1)

generation_number=1
plt.figure()
while generation_number< 50:
    population=GeneticAlgorithm().evolve(population)
    population.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
    plt.scatter(generation_number,population.get_chromosomes()[0].get_fitness() )
    _print_population(population,generation_number)
    plt.pause(0.01)
    generation_number+=1
print('Minmization -- at the function :',population.get_chromosomes()[0])
plt.show()


