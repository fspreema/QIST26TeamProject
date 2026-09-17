from wcnf_matrix import * 
import numpy as np

def ising_partition_function_wmc(n_spins, interaction, external_field, beta):
    """
    interaction: dict {(i, j): J_ij} para cada par que interactúa (i < j)
    external_field: lista [h_0, h_1, ..., h_{n-1}]
    beta: inverso de la temperatura
    Devuelve Z_beta calculado vía Weighted Model Counting.
    """
    # Una variable booleana por spin
    spin_vars = [BoolVar() for _ in range(n_spins)]

    # Una variable auxiliar booleana por cada interacción (i,j)
    interactions = list(interaction.items())  # [((i,j), J_ij), ...]
    aux_vars = [BoolVar() for _ in interactions]

    clauses = []
    for ((i, j), _), aux in zip(interactions, aux_vars):
        vi, vj = spin_vars[i], spin_vars[j]
        # aux <-> (vi <-> vj)   codificado en 4 cláusulas CNF:
        clauses.append([vi, -vj, -aux])
        clauses.append([vi, vj, aux])
        clauses.append([-vi, vj, -aux])
        clauses.append([-vi, -vj, aux])

    cnf = CNF(clauses)

    all_vars = spin_vars + aux_vars
    weight_func = WeightFunction(all_vars)

    # Pesos de las variables de spin: True=+1, False=-1
    for var, h in zip(spin_vars, external_field):
        weight_func[var, True] = np.exp(beta * h)
        weight_func[var, False] = np.exp(-beta * h)

    # Pesos de las variables auxiliares de interacción
    for ((_, _), J), aux in zip(interactions, aux_vars):
        weight_func[aux, True] = np.exp(beta * J)
        weight_func[aux, False] = np.exp(-beta * J)

    counter = DPMC()
    result = counter.model_count(cnf, weight_func)
    return result.model_count


# --- Ejemplo: n=3, J=1.5 constante (vecinos), h=0.5 constante ---
n = 3
J = 1.5
h = 0.5
interaction = {(0, 1): J, (1, 2): J}   # solo vecinos
external_field = [h, h, h]
beta = 1.0

Z = ising_partition_function_wmc(n, interaction, external_field, beta)
print("Z_beta (WMC) =", Z)