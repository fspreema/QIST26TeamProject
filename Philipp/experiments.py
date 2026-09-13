from itertools import product
import networkx as nx

import numpy as np
import matplotlib.pyplot as plt
from IsingModel import LatticeIsing





betas = np.linspace(0, 10, 20)
values = []
for beta in betas:
    cnf, WeightFunction = (
        LatticeIsing(
            dimensions=2,
            size=2,
            weights=0.1,#This gives lattice on a torus with homogenous weights 0.1
            dimsubspace=2
            ).partition_function(beta)
        )

    values.append(WeightFunction(cnf))
    print("current beta:", beta, "   Z(beta):", values[-1])

# Plot
plt.figure(figsize=(6, 4))
plt.plot(betas, values)
plt.xlabel(r"$\beta$")
plt.ylabel(r"$Z(\beta)$")
plt.title("Partition Function vs. $\\beta$")
plt.grid(True)
plt.show()
cnf, WeightFunction = LatticeIsing(dimensions = 2, size = 2, weights = 0.1, dimsubspace = 2).partition_function(1)
print(WeightFunction(cnf))
values = []



betas = -np.linspace(0, 1, 20)
values = []

size = 6
weights = {}
for i in range(size**2):
    if i + size < size*size:
        weights[(i,i+size)]  = np.random.normal()
    if i%size != size -1:
        weights[(i,i+1)] = np.random.normal()




np.random.seed(57)
# This will build a square lattice with normally distributed nearest neighbor weigths  
# Build graph
G = nx.Graph()

# Node positions on square lattice
pos = {
    i: (i % size, -(i // size))
    for i in range(size**2)
}

G.add_nodes_from(pos)

for (i, j), w in weights.items():
    G.add_edge(i, j, weight=w)

# Extract edge weights
edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

# Plot
plt.figure(figsize=(6, 6))

nx.draw_networkx_nodes(
    G, pos,
    node_color="lightgray",
    node_size=500
)

edges = nx.draw_networkx_edges(
    G, pos,
    edge_color=edge_weights,
    edge_cmap=plt.cm.coolwarm,
    width=3
)

nx.draw_networkx_labels(G, pos)

plt.colorbar(edges, label="weight")
plt.axis("equal")
plt.axis("off")
plt.title("Weighted square lattice")
plt.show()
for beta in betas:
    cnf, WeightFunction = (
        LatticeIsing(
            dimensions=2,
            size=size,
            weights=weights,
            dimsubspace=2
            ).partition_function(beta)
        )

    values.append(WeightFunction(cnf))
    print("current beta:", beta, "   log(Z(beta)):", np.log(values[-1]))

# Plot
plt.figure(figsize=(6, 4))
plt.plot(betas, np.log(values))
plt.xlabel(r"$\beta$")
plt.ylabel(r"$log(Z(\beta))$")
plt.title("Partition Function vs. $\\beta$")
plt.grid(True)
plt.show()
cnf, WeightFunction = LatticeIsing(dimensions = 2, size = 2, weights = 0.1, dimsubspace = 2).partition_function(1)
print(WeightFunction(cnf))