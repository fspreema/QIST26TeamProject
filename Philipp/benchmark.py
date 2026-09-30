from itertools import product
import time
from wcnf_matrix import *

import numpy as np
import matplotlib.pyplot as plt
from IsingModel import LatticeIsing
from TransIsing import LatticeTransIsing

solvers = {'DPMC': DPMC, 'Cachet':Cachet, 'TensorOrder':TensorOrder}
Results = {solver : [] for solver in solvers}
Energy = Results
np.random.seed(57)
for size in range(2,6):
    
    weights = {}
    
    for i in range(size**2):
        if i + size < size*size:
            weights[(i,i+size)]  = np.random.normal()
        if i%size != size -1:
            weights[(i,i+1)] = np.random.normal()
    
    
    exclude = None
    for solver in solvers:
        set_model_counter(solvers[solver])
        try:
            t_0 = time.perf_counter()
            cnf, WeightFunction = (
                                    LatticeIsing(
                                        dimensions=2,
                                        size=size,
                                        weights=weights,
                                        dimsubspace=2
                                        ).partition_function(1)
                                    )
            free_energ = np.log(WeightFunction(cnf))
            t_1 = time.perf_counter()
            print('Solver = ', solver, 'size = ', size, 'Free Energy = ', free_energ, 'time = ', t_1 - t_0 )
            Energy[solver].append(free_energ)
            Results[solver].append(t_1 -t_0)
        except RuntimeError:
            Results[solver].append(None)
            print("Run time Error occurred, {name}, {size}".format( name = solver, size = size))
            exclude = solver

    if exclude != None:
        solvers.pop(exclude)

size = 10
weights = {}
for i in range(size**2):
        if i + size < size*size:
            weights[(i,i+size)]  = np.random.normal()
        if i%size != size -1:
            weights[(i,i+1)] = np.random.normal()

cnf, WeightFunction = (
                                    LatticeIsing(
                                        dimensions=2,
                                        size=size,
                                        weights=weights,
                                        dimsubspace=2
                                        ).partition_function(0.1)
                                    )
print(WeightFunction(cnf))
