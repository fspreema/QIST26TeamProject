from Tenpy_code.RandomBondIsing import RandomBondIsing as TenpyRBI
from datetime import datetime
from RBIM_model import RandomBondIsingModel, BaseModel
from BenchMark import BenchMark
from wcnf_matrix import ModelCounter, DPMC, Cachet, TensorOrder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    # === Parameters here ===   
    save_path = r"Results\Tenpy_model"
    solver = Cachet
    max_length = 16
    p = 0.5
    beta = 0.5
    average = 16

    # Prepare variables
    filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), save_path)
    bm = BenchMark(RandomBondIsingModel, solver)
    sizes = [int(i) for i in range( 3, max_length +1 )]

    datetime_string = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    filename = rf"{solver.__name__}_MaxLength-{max_length}_Average-{average}_{datetime_string}"

    # Triangle Torus
    model_params = {
        "lattice": "Triangular",
        "p": p,
        "bc_x": "periodic",
        "bc_y": "periodic",
    }
    shapes = [ (s,s) for s in sizes ]
    TT_runtimes, TT_stds = bm.tenpy_runtime_vs_shape_benchMark(TenpyRBI, model_params, shapes, average, skip_on_fail = True, beta = beta)

    # Square Cilinder
    model_params = {
        "lattice": "Square",
        "p": p,
        "bc_x": "periodic",
        "bc_y": "open",
    }
    shapes = [ ((s*s)//3, 3) for s in sizes ]
    SC_runtimes, SC_stds = bm.tenpy_runtime_vs_shape_benchMark(TenpyRBI, model_params, shapes, average, skip_on_fail = True, beta = beta)

    # Square Open
    model_params = {
        "lattice": "Square",
        "p": p,
        "bc_x": "open",
        "bc_y": "open",
    }
    shapes = [ (s,s) for s in sizes ]
    SO_runtimes, SO_stds = bm.tenpy_runtime_vs_shape_benchMark(TenpyRBI, model_params, shapes, average, skip_on_fail = True, beta = beta)

    # Square Torus
    model_params = {
        "lattice": "Square",
        "p": p,
        "bc_x": "periodic",
        "bc_y": "periodic",
    }
    shapes = [ (s,s) for s in sizes ]
    ST_runtimes, ST_stds = bm.tenpy_runtime_vs_shape_benchMark(TenpyRBI, model_params, shapes, average, skip_on_fail = True, beta = beta)
    

    # save data to csv file
    df = pd.DataFrame({'Size':sizes,
                       "SC_runtimes": SC_runtimes, "SC_stds": SC_stds,
                       "SO_runtimes": SO_runtimes, "SO_stds": SO_stds,
                       "ST_runtimes": ST_runtimes, "ST_stds": ST_stds,
                       "TT_runtimes": TT_runtimes, "TT_stds": TT_stds })
    df.to_csv(os.path.join(filepath, filename + ".csv"))

    # plot results
    fig, ax = plt.subplots()
    ax.plot(sizes, SC_runtimes, color = "tab:green", marker = "o", linestyle = "-", label = r"Square - cylinder ($L_x \times 3$)")
    ax.plot(sizes, SO_runtimes, color = "tab:blue", marker = "o", linestyle = "-", label = r"Square - open ($L_x \times L_x$)")
    ax.plot(sizes, ST_runtimes, color = "tab:orange", marker = "s", linestyle = "-", markersize = 9, markerfacecolor = "none", zorder = 3, label = r"Square - torus ($L_x \times L_x$)")
    ax.plot(sizes, TT_runtimes, color = "tab:red", marker = "^", linestyle = "--", markersize = 5, zorder = 4, label = r"Triangular - torus ($L_x \times L_x$)")
    ax.set_xlabel("Length $L_x$")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Runtime for Z")
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(filepath, filename + ".pdf"))
    plt.show()


