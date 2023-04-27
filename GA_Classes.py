import numpy as np
import random

class Genome():
    def __init__(self,Genome_Markers,Genome_Domains,new_genome=None,initial=False):
        # print(Genome_Domains)
        self.genome={key:[] for key in Genome_Domains}
        self.Genome_Domains=Genome_Domains
        self.Genome_Markers=Genome_Markers
        if new_genome==None:
            for marker in Genome_Markers:
                if isinstance(Genome_Domains[marker],list):
                    self.genome[marker]+=[np.random.choice(Genome_Domains[marker])]
                elif initial==False or Genome_Domains[marker][3]==None:
                    if Genome_Domains[marker][1]==int:
                        self.genome[marker]+=[random.randint(*Genome_Domains[marker][0])]
                    elif Genome_Domains[marker][1]==float:
                        self.genome[marker]+=[random.random()*(Genome_Domains[marker][0][1]-Genome_Domains[marker][0][0])+Genome_Domains[marker][0][0]]
                else:
                    if Genome_Domains[marker][1]==int:
                        self.genome[marker]+=[random.randint(*Genome_Domains[marker][3])]
                    elif Genome_Domains[marker][1]==float:
                        self.genome[marker]+=[random.random()*(Genome_Domains[marker][3][1]-Genome_Domains[marker][3][0])+Genome_Domains[marker][3][0]]
        else:
            self.genome=new_genome

        # print(self.genome)

    def copy(self):
        new_genome={key:[] for key in self.Genome_Domains}
        for marker in self.Genome_Markers:
            new_genome[marker]+=[self.genome[marker]]
        return Genome(new_genome)

    def __eq__(self, other):
        return False not in [g==other.genome[k][i] for k in self.genome for i,g in enumerate(self.genome[k])]
    
    def __repr__(self):
        return str(self.genome)
    
    def __getitem__(self, key):
        return self.genome[key]

class Fitness():
    def __init__(self, fitness):
        if isinstance(fitness,list):
            self._fitness = fitness
        elif isinstance(fitness,tuple):
            self._fitness = list(fitness)
        else:
            self._fitness = [fitness]

    def __lt__(self, other):
        lt=None
        for sf,of in zip(self._fitness,other._fitness):
            if sf>of:
                return False
            if sf<of:
                lt=True
        if lt==None:
            return False
        return True

    def __gt__(self, other):
        gt=None
        for sf,of in zip(self._fitness,other._fitness):
            if sf<of:
                return False
            if sf>of:
                gt=True
        if gt==None:
            return False
        return True

    def __eq__(self, other):
        return not self.__lt__(other) and not self.__gt__(other)
    
    def __repr__(self):
        #print(self._fitness)
        return ' '.join(list(map(str,self._fitness)))

    def __getitem__(self, key):
        return self._fitness[key]

# class Fitness_Replicate(Fitness):
#     def __init__(self, fitness):
#         if isinstance(fitness,list):
#             self._fitness = fitness
#         else:
#             self._fitness = [fitness]
    
#     def

class Individual(Genome):
    def __init__(self, fitness_fcn,Genome_Markers,Genome_Domains, ID, new_genome=None,initial=False):
        # print(Genome_Markers,Genome_Domains)
        Genome.__init__(self,Genome_Markers,Genome_Domains,new_genome,initial)

        # print(self.genome)
        
        self._ID = ID[0]
        ID[0]+=1

        fitness,violation = fitness_fcn(self.genome,self._ID,0)
        #Don't Try To Save The Fitness Fucntion!!!
        self.fitness = Fitness(fitness)
        
        self.violation = violation
        self.rank=0
        self.replicates=1
        self.fitness_hist=[self.fitness]
        #self.replicate(fitness_fcn)

    def getID(self):
        return self._ID
    
    def __repr__(self):
        return str(self.genome) + " " + str(self.fitness)

    def __eq__(self, other):
        return (False not in [f==other.fitness._fitness[i] for i,f in enumerate(self.fitness._fitness)]) or Genome.__eq__(self,other)

    def replicate(self,fitness_fcn):
        if self.replicates>=6: return
        expected_sd=[0.001, 0.01, 0.5642839364293354, 0.7252013499815404, 0.7975273470064533, 0.8402642582051366, 0.868401850146515, 0.8881477704288123, 0.9037605415930194, 0.9143872906413865, 0.9226680183514016]
        new_fitness,new_violation = fitness_fcn(self.genome,self._ID,self.replicates)
        self.fitness_hist+=[new_fitness]
        for i in range(len(self.fitness._fitness)):
            self.fitness._fitness[i]=np.average([f[i] for f in self.fitness_hist])
        self.violation=(self.violation*self.replicates+new_violation)/(self.replicates+1)
        self.replicates+=1
        # self.fitness._fitness[1]=np.std([f[0] for f in self.fitness_hist])/expected_sd[self.replicates]**1.35
