import numpy as np
import timeit

number_in_range = 10000
numbe_of_executions = 1000

def test():
    for x in np.linspace(0.0000001, 9999999, number_in_range):
        y = np.log(x)

result = timeit.timeit("test()", number = numbe_of_executions, setup="import numpy as np; from __main__ import test")
print(f"execution time for np.log(x) : {result/(number_in_range*numbe_of_executions):.2e} s")