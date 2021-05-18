from matplotlib import markers
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import random
import pandas as pd
import matplotlib.animation as animation
import os
import progressbar
from time import sleep
class GeneticAlgorithm_Base:
    """
    firstly defined the vvariable which to be saved.
    """
    @classmethod
    def _get_userInputs(cls,f,dim=1,population_size=500,max_intr=1,xlb=[-200,-2,0,-2],
    xUb=[200,2,1,2],Tournament_Selection_Size=2,MUTATION_RATE=0.25,
    mutant_sigma=[0.5,0.05,0.05,0.05],blx_alpha=0.5,NUMBER_OF_ELITE_CHROMOSOMES=1
    ,dir_name_save="E:\Work\Code\Hybrid_sensors_model"):
        cls.population_size=population_size
        cls.max_intr=max_intr
        cls.xLowerList=xlb
        cls.xUpperList=xUb
        cls.Tournament_Selection_Size=Tournament_Selection_Size
        cls.MUTATION_RATE=0.25
        cls.mutant_sigma=mutant_sigma
        cls.blx_alpha=blx_alpha
        cls.NUMBER_OF_ELITE_CHROMOSOMES=NUMBER_OF_ELITE_CHROMOSOMES
        cls.Pc=0.9
        cls.dim=dim
        cls.f=f
        cls.dir_name=dir_name_save
        
# Generate the Chromosome Class
class Chromosome:
    def __init__(self):
        self._genes=[]
        self._fitness=0
        for i in range(GeneticAlgorithm_Base.dim):
            a=np.random.uniform(low=GeneticAlgorithm_Base.xLowerList[i], high=GeneticAlgorithm_Base.xUpperList[i])
            self._genes.append(a)

    def get_fitness(self):
        self._fitness=GeneticAlgorithm_Base.f(self._genes)#GeneticAlgorithm_Base.get_f(self._genes)#100*(self._genes[1]-self._genes[0]**2)**2+(1-self._genes[0])**2 
        return self._fitness

    def get_genes(self):
        return self._genes
    
    def __str__(self):
        return self._genes.__str__()
# Generate the Population Class
class Population:
    def __init__(self,size):
        self._chromosomes=[]
        i=0
        while i< size:
            self._chromosomes.append(Chromosome())
            i+=1
    def get_chromosomes(self):return self._chromosomes

# Main Class of the Genetic Algorithm
class GeneticAlgorithm:
    @staticmethod
    ## Evolve the Population
    def evolve(pop):
        # return GeneticAlgorithm._crossover_population(pop)
        newPop=GeneticAlgorithm._mutate_population(GeneticAlgorithm._crossover_population(pop))
        return GeneticAlgorithm._survivorStage(pop,newPop)
    
    @staticmethod # selection process
    def _select_tournament_population(pop):
        tournament_pop=Population(0)
        i=0
        while i < GeneticAlgorithm_Base.Tournament_Selection_Size:
            tournament_pop.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0,GeneticAlgorithm_Base.population_size)])
            i+=1
        tournament_pop.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
        return tournament_pop
    
    
    #creating the Crossover Population:
    @staticmethod
    def _crossover_population(pop):
        crossover_pop=Population(0) # HERE I DEFINED the empty chromosomes when put the zero
        for i in range(GeneticAlgorithm_Base.NUMBER_OF_ELITE_CHROMOSOMES):
            crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i]) ## we will move this chromose to the next gen
        i=GeneticAlgorithm_Base.NUMBER_OF_ELITE_CHROMOSOMES
        ## here we will exclude the elite chromsome
        while i < GeneticAlgorithm_Base.population_size:
            # crossover population will have population after the selection.Then we select the best 2 Chromosome
            chromosome1=GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0] # here we call to the _tournament_population
            chromosome2=GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0]
            crossover_pop.get_chromosomes().append(GeneticAlgorithm._blx_alpha_crossover(chromosome1, chromosome2))
            i+=1
        return crossover_pop
    #creating the Mutation Population:
    @staticmethod
    def _mutate_population(pop):
        for i in range(GeneticAlgorithm_Base.NUMBER_OF_ELITE_CHROMOSOMES,GeneticAlgorithm_Base.population_size):# EXCLUDE THE ELITE CHROMOSOME
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
            gamma=(1+2*GeneticAlgorithm_Base.blx_alpha)*u-GeneticAlgorithm_Base.blx_alpha
            crossover_chrom.get_genes()[i]=p1*(1-gamma)+gamma*p2
            # print(crossover_chrom.get_genes()[i])
        return crossover_chrom
            

    @staticmethod
    def _Normaldistribution_mutate_chromosome(chromosome):
        for i in range(chromosome.get_genes().__len__()):
            chromosome.get_genes()[i]=chromosome.get_genes()[i]+np.random.normal(0, GeneticAlgorithm_Base.mutant_sigma[i])
            if chromosome.get_genes()[i] > GeneticAlgorithm_Base.xUpperList[i]:
                chromosome.get_genes()[i]=GeneticAlgorithm_Base.xUpperList[i]
            if chromosome.get_genes()[i] < GeneticAlgorithm_Base.xLowerList[i]:
                chromosome.get_genes()[i]=GeneticAlgorithm_Base.xLowerList[i]
    
    @staticmethod
    def _survivorStage(oldpop,newpop):
        survivalPop=Population(0)
        oldpop.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
        newpop.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
        n=GeneticAlgorithm_Base.population_size//2
        i=0
        while i < n:
            survivalPop.get_chromosomes().append(oldpop.get_chromosomes()[i])
            i+=1
        j=0
        while j < n:
            survivalPop.get_chromosomes().append(newpop.get_chromosomes()[j])
            j+=1
        return survivalPop

class GA_strat:
    def __init__(self,ifsaveReport=False):
        self.ifsaveReport=ifsaveReport
    
    def inital_population(n):
        # Inital Population
        population=Population(GeneticAlgorithm_Base.population_size)
        # Sorting the Initial Population
        population.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
        # Priniting the Initial Population
        self._print_population(population,0)


    def RUN(self):
        # Inital Population
        population=Population(GeneticAlgorithm_Base.population_size)
        # Sorting the Initial Population
        population.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
        # Priniting the Initial Population
        # self._print_population(population,0)
        
        generation_number=1
        count=[]
        count.append(generation_number)
        plt.figure()
        ax=plt.gca()
        bar=self.bar_notation()
        # line, = ax.plot([], [], lw=2)
        while generation_number< GeneticAlgorithm_Base.max_intr:
            population=GeneticAlgorithm().evolve(population)
            population.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=False)
            #--------------------------------saving the data-------------------------
            #----------------
            # line=self.setting_line(line)
            ax.scatter(generation_number,population.get_chromosomes()[0].get_fitness(), marker='o', c='k')
            ax.set_xlabel('# Iter')
            ax.set_ylabel('Fitness Value')
            ax.set_title('Convergance Plot')
            # self._print_population(population,generation_number)
            plt.pause(0.01)
            # print('Minmization -- at the function :',population.get_chromosomes()[0])
            generation_number+=1
            count.append(generation_number)
            bar.update(generation_number)
            # sleep(0.1)
            # np.save('Best_solution.csv',[population.get_chromosomes()[0],])
        bar.finish()
        print('Minmization -- at the function :',population.get_chromosomes()[0])
        print('----------------------------Finshed---------------------------------------')
        if self.ifsaveReport:
            os.mkdir(GeneticAlgorithm_Base.dir_name)
            report={'Fitness':[], 'X0':[],'X1':[],'X2':[],'X3':[]}
            report['Fitness'].append(population.get_chromosomes()[0].get_fitness())
            for No_var,Varb in enumerate(np.array(population.get_chromosomes()[0].get_genes())):
                report['X'+str(No_var)].append((Varb))
            data_best=pd.DataFrame.from_dict(report)
            data_best.to_csv(GeneticAlgorithm_Base.dir_name+'\\'+'Best_solution.csv')
            plt.savefig(GeneticAlgorithm_Base.dir_name+'\\'+'Convergence.png')
            print('-----------Report Finished---------------------------------------------')
    def setting_line(self,line):
        line.set_data([], [])
        return line,
    
    def _print_population(self,pop, gen_number):
        print('\n------------------------------------------------')
        print("Generation #", gen_number, "|Fittest Chromosome fitness :", pop.get_chromosomes()[0].get_fitness())
        print("---------------------------------------------------")
        i=0
        for x in pop.get_chromosomes():
            print("Chromosome #",i,":", x, "| fitness :", x.get_fitness())
            i+=1

    def bar_notation(self):
        bar = progressbar.ProgressBar(maxval=GeneticAlgorithm_Base.max_intr, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        return bar