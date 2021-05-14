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
import Rastrigin_function_opt as RF
def objective_function(f):
    return f

class GeneticAlgorithm_parameters:
    def __init__(self,dim=2,population_size=100,max_intr=50,xlb=[-100,-100,0],
    xUb=[100,100,1],Tournament_Selection_Size=2,MUTATION_RATE=0.25,mutant_sigma=[0.5,0.1,0.1,0.1],blx_alpha=0.5):
        # self.generation_number=generation_number
        self.population_size=population_size # is  should be in even
        self.max_intr=max_intr
        self.xLowerList=xlb
        self.xUpperList=xUb
        self.Tournament_Selection_Size=Tournament_Selection_Size
        self.MUTATION_RATE=0.25
        self.mutant_sigma=mutant_sigma
        self.blx_alpha=blx_alpha
        self.NUMBER_OF_ELITE_CHROMOSOMES=1
        self.dim=dim
        self.__fun=0
        
    def get_f(self,x):
        return self.__fun(x)

    def set_f(self,fun):
        return self.__fun

class Chromosome:
    def __init__(self):
        self._genes=[]
        self._fitness=0
        for i in range(GeneticAlgorithm_parameters().dim):
            a=np.random.uniform(low=GeneticAlgorithm_parameters().xLowerList[i], high=GeneticAlgorithm_parameters().xUpperList[i])
            self._genes.append(a)

    def get_fitness(self):
        # GeneticAlgorithm_parameters().get_fitness_function
        self._fitness=RF.f(self._genes)#GeneticAlgorithm_parameters().get_f(self._genes)#100*(self._genes[1]-self._genes[0]**2)**2+(1-self._genes[0])**2 
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
    
    @staticmethod # selection process
    def _select_tournament_population(pop):
        tournament_pop=Population(0)
        i=0
        while i < GeneticAlgorithm_parameters().Tournament_Selection_Size:
            tournament_pop.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0,GeneticAlgorithm_parameters().population_size)])
            i+=1
        tournament_pop.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
        return tournament_pop
    
    
    #creating the Crossover Population:
    @staticmethod
    def _crossover_population(pop):
        crossover_pop=Population(0) # HERE I DEFINED the empty chromosomes when put the zero
        for i in range(GeneticAlgorithm_parameters().NUMBER_OF_ELITE_CHROMOSOMES):
            crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i]) ## we will move this chromose to the next gen
        i=GeneticAlgorithm_parameters().NUMBER_OF_ELITE_CHROMOSOMES
        ## here we will exclude the elite chromsome
        while i < GeneticAlgorithm_parameters().population_size:
            # crossover population will have population after the selection.Then we select the best 2 Chromosome
            chromosome1=GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0] # here we call to the _tournament_population
            chromosome2=GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0]
            crossover_pop.get_chromosomes().append(GeneticAlgorithm._blx_alpha_crossover(chromosome1, chromosome2))
            i+=1
        return crossover_pop
    #creating the Mutation Population:
    @staticmethod
    def _mutate_population(pop):
        for i in range(GeneticAlgorithm_parameters().NUMBER_OF_ELITE_CHROMOSOMES,GeneticAlgorithm_parameters().population_size):# EXCLUDE THE ELITE CHROMOSOME
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
            gamma=(1+2*GeneticAlgorithm_parameters().blx_alpha)*u-GeneticAlgorithm_parameters().blx_alpha
            crossover_chrom.get_genes()[i]=p1*(1-gamma)+gamma*p2
            # print(crossover_chrom.get_genes()[i])
        return crossover_chrom
            

    @staticmethod
    def _Normaldistribution_mutate_chromosome(chromosome):
        for i in range(chromosome.get_genes().__len__()):
            chromosome.get_genes()[i]=chromosome.get_genes()[i]+np.random.normal(0, GeneticAlgorithm_parameters().mutant_sigma[i])
            if chromosome.get_genes()[i] > GeneticAlgorithm_parameters().xUpperList[i]:
                chromosome.get_genes()[i]=GeneticAlgorithm_parameters().xUpperList[i]
            if chromosome.get_genes()[i] < GeneticAlgorithm_parameters().xLowerList[i]:
                chromosome.get_genes()[i]=GeneticAlgorithm_parameters().xLowerList[i]
    
    @staticmethod
    def _survivorStage(oldpop,newpop):
        survivalPop=Population(0)
        oldpop.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
        newpop.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
        n=GeneticAlgorithm_parameters().population_size//2
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

def GA_RUN():
    population=Population(GeneticAlgorithm_parameters().population_size)
    population.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
    _print_population(population,0)

    generation_number=1
    plt.figure()
    while generation_number< GeneticAlgorithm_parameters().max_intr:
        population=GeneticAlgorithm().evolve(population)
        population.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
        plt.scatter(generation_number,population.get_chromosomes()[0].get_fitness() )
        _print_population(population,generation_number)
        plt.pause(0.01)
        generation_number+=1
    print('Minmization -- at the function :',population.get_chromosomes()[0])
    plt.show()
if __name__=='__main__':
    GA1=GeneticAlgorithm_parameters()
    GA_RUN()

