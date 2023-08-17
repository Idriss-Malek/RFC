import pandas as pd
import numpy as np
import random
from time import time

from ..structs.ensemble import Ensemble
from ..structs.utils import idenumerate


class Genetic:
    ensemble : Ensemble
    dataset : pd.DataFrame

    def __init__(self, ensemble, dataset) -> None:
        self.ensemble = ensemble
        self.dataset = dataset
        self.n_trees = len(ensemble)
        self.selection = self.initial_selection()
        self.scores = {tuple(u) : self.fitness(u) for u in self.selection}
        self.current_scores = np.apply_along_axis(lambda row: self.scores[tuple(row)], axis=1, arr=self.selection)
        self.best = 0
        self.unchange = 0
        i = 0
        for score in self.current_scores:
            if self.best < score:
                self.best = score
                self.u = self.selection[i]
            i += 1

    def check(self,u,x):
        return self.ensemble.klass(x,u) == self.ensemble.klass(x)

    def fitness(self,u):
        if (u == np.zeros(len(self.ensemble))).all():
            return 0
        if not self.dataset.apply(lambda x : self.check(u,x), axis=1).all():
            return 0
        return 101 - sum(u)

    def crossover(self, u1, u2):
        cross_point = random.randint(1,self.n_trees - 1)
        u = np.concatenate([u1[:cross_point], u2[cross_point:]])
        return u
    
    def mutation(self, u):
        mutation_point = random.randint(0,self.n_trees - 1)
        u[mutation_point] = 1 - u[mutation_point]
        if tuple(u) not in self.scores:
            self.scores[tuple(u)] = self.fitness(u)

    def initial_selection(self):
        return np.vstack((np.ones(self.n_trees),np.random.randint(2, size=(8, self.n_trees))))
    
    def next_gen(self):
        
        big3 = np.argpartition(self.current_scores, -3)[-3:]
        new_gen = np.empty((9,self.n_trees))
        i = 0
        for index in big3:
            new_gen[i] = self.selection[index]
            i += 1
        for k in range(2):
            for j in range(k + 1,3):
                new_gen[i] = self.crossover(new_gen[k],new_gen[j])
                new_gen[i + 1] = self.crossover(new_gen[j],new_gen[k])
                i += 2
        for i in range(9):
            self.mutation(new_gen[i])
        self.selection = new_gen
        self.current_scores = np.apply_along_axis(lambda row: self.scores[tuple(row)], axis=1, arr=self.selection)
        i = 0
        for score in self.current_scores:
            if self.best < score:
                self.best = score
                self.u = self.selection[i]
                self.unchange = 0
            i += 1
                
        self.unchange = 1
    
    def genetic(self, iterations = 100):
        for iteration in range(iterations):
            t =time()
            if self.best == 100:
                with open('terminal_genetic.csv', 'a+') as f:
                    f.write(f'{iteration} : {time() - t}, 1 tree found')
                return
            self.next_gen()
            with open('terminal_genetic.csv', 'a+') as f:
                    f.write(f'{iteration} : {time() - t} , {self.best}\n')
            if self.unchange == 100:
                return
        return
            
    


            






        
