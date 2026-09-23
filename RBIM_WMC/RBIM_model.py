from wcnf_matrix import *
import numpy as np
import matplotlib.pyplot as plt


class RBIM_class:
    x: int

    def __init__(self):
        self.x = 69

    def print_it(self):
        print(self.x)

    def print_exp(self):
        print(np.exp(self.x))