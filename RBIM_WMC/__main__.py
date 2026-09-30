from RBIM_model import RandomBondIsingModel
from BenchMark import BenchMark
from wcnf_matrix import ModelCounter, DPMC, Cachet, TensorOrder
import time
import numpy as np
import matplotlib.pyplot as plt

SOLVERS: tuple[type[ModelCounter], ...] = (DPMC, Cachet, TensorOrder,)  

def shape_vs_runtime(max_size : tuple[int,int]= (3,3), solver : ModelCounter = TensorOrder):
    bm = BenchMark(RandomBondIsingModel, solver)

    shapes = []
    for y in range(1,max_size[1]+1):
        for x in range(1,max_size[0]+1):
            if (x == 1 and y == 1):
                continue
            shapes.append((x,y))

    runtimes = []
    for shape in shapes:
        models = [RandomBondIsingModel.generate_lattice(shape, seed=i) for i in range(8)]
        measured_runtimes, _, _ = bm.run_solver(models, 1.0)
        runtimes.append(np.average(measured_runtimes))
        print(f"shape = {shape} -> {runtimes[-1]} s")
    runtimes.insert(0, 0)

    z = np.asarray(runtimes).reshape((max_size[1],max_size[0]))

    fig, ax = plt.subplots()

    image = ax.imshow(z,aspect="equal",cmap="viridis", extent=[0.5,max_size[0]+0.5,0.5,max_size[1]+0.5])
    ax.set_xlabel("lattice size x")
    ax.set_ylabel("lattice size y")
    ax.set_title("runtime by lattice size")
    ax.set_xticks(np.arange(1,max_size[0]+1))
    ax.set_yticks(np.arange(1,max_size[1]+1))
    fig.colorbar(image, ax=ax, label="runtime [s]")

    plt.tight_layout()

    plt.show()

def magnetization_vs_beta(shape: tuple[int,int], betas: list, p_ferro: float = 1.0):
    bm = BenchMark(RandomBondIsingModel, DPMC)

    start_1 = time.perf_counter()
    bm.magnetization_vs_beta_benchMark(shape=shape, betas=betas, p_ferro=p_ferro)
    end_1 = time.perf_counter()
    time_mag = end_1 - start_1

    start_2 = time.perf_counter()
    bm.magnetization_squared_vs_beta_benchMark(shape=shape, betas=betas, p_ferro=p_ferro)
    end_2 = time.perf_counter()
    time_mag2 = end_2 - start_2

    print("\n" + "="*40)
    print("        Results        ")
    print("="*40)
    print(f"Time Magnetizaiton:   {time_mag:.4f} seconds")
    print(f"Time Magnetization Squared:   {time_mag2:.4f} seconds")


if __name__ == "__main__":

    shape_vs_runtime((12,12), Cachet)

    L = 2
    magnetization_vs_beta(shape=(L, L), betas=np.linspace(0.0, 3.0, 10), p_ferro=1.0)

    # sides = np.arange(2,8,2)
    # shapes = [(int(n),int(n)) for n in sides]
    # bm.runtime_vs_lattice_benchMark(shapes, average = 4)

    # # calculates error for random bond chain
    # sides = np.arange(2,12,1)
    # shapes = [(int(n),1) for n in sides]
    # bm.error_vs_lattice_benchMark(shapes, average = 32)

