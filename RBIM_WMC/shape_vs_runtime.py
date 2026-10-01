from RBIM_model import RandomBondIsingModel, BaseModel
from BenchMark import BenchMark
from wcnf_matrix import ModelCounter, DPMC, Cachet, TensorOrder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def shape_vs_runtime(max_size : tuple[int,int]= (3,3), solver : ModelCounter = TensorOrder, savepath: str = ".", model: BaseModel = RandomBondIsingModel):
    bm = BenchMark(model, solver)
    filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), savepath)

    shapes = []
    for y in range(1,max_size[1]+1):
        for x in range(1,max_size[0]+1):
            if (x == 1 and y == 1):
                continue
            shapes.append((x,y))

    runtimes = bm.runtime_vs_lattice_benchMark(shapes, average = 16)
    shapes.insert(0,(1,1))
    runtimes.insert(0, 0)

    # save data to .csv file
    df = pd.DataFrame({'Shapes':shapes, "Runtimes": runtimes}) #pd.DataFrame([shapes, runtimes], columns=["Shapes", "Runtimes"])
    df.to_csv(os.path.join(filepath, rf"shape_vs_runtime_({shapes[-1][0]} {shapes[-1][1]})_{solver.__name__}_{model.__name__}.csv"))

    # plot data as image
    z = np.asarray(runtimes).reshape((max_size[1],max_size[0]))

    fig, ax = plt.subplots()

    image = ax.imshow(z,aspect="equal",cmap="viridis", extent=[0.5,max_size[0]+0.5,0.5,max_size[1]+0.5], origin='lower')
    ax.set_xlabel("lattice size x")
    ax.set_ylabel("lattice size y")
    ax.set_title("runtime by lattice size")
    ax.set_xticks(np.arange(1,max_size[0]+1))
    ax.set_yticks(np.arange(1,max_size[1]+1))
    fig.colorbar(image, ax=ax, label="runtime [s]")

    plt.tight_layout()
    plt.savefig(os.path.join(filepath, rf"shape_vs_runtime_({shapes[-1][0]} {shapes[-1][1]})_{solver.__name__}_{model.__name__}.pdf"))
    plt.show()


if __name__ == "__main__":

    max_size = (4,4)
    solver = TensorOrder
    save_path = r"Results\Shape_vs_runtime"
    model = RandomBondIsingModel

    shape_vs_runtime(max_size, solver, save_path, model)
