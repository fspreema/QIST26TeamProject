import numpy as np
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfSite


class RandomBondIsing(CouplingMPOModel):

    """
    Implements the random bond ising model on a 2D square lattice with nearest neighbor interactions. 
    -> The Hamiltonian is given by: H = - sum_{<i,j>} J_{ij} S^z_i S^z_j
    -> The coupling strengths J_{ij} are drawn from J_{ij} = p * delta(j - 1) + (1 - p) * delta(j + 1)
       where p is the probability of having a ferromagnetic bond
    """

    # DO NOT override the __init__ method, as it is already implemented in the CouplingMPOModel class!!!
 
    # Override Internal Method to initialize the sites
    # -> Although model_params is not used in this method, it is still required to be 
    # passed as an argument due to the method signature in the parent class!!
    def init_sites(self, model_params:dict) -> SpinHalfSite:
        # Since H = - sum_{<i,j>} J_{ij} S^z_i S^z_j, we can use SpinHalfSite with conserve='Sz'
        # -> This is due to the fact that h=0
        return SpinHalfSite(conserve='Sz')

    def get_coupling_strength(self, model_params:dict, u1, u2, dx) -> float:
        """
        Returns an array of coupling strengths J
        -> Since neighbour loop is over all verticial and then horizontal family members,
           each memebr needs their own coupling strength.
        -> Given with an array where the individual J terms are stored

                 ##### Horizontal #####
                o -- J00 -- o -- J10 -- o
                |           |           |
                o -- J01 -- o -- J11 -- o
                |           |           |
                o -- J02 -- o -- J12 -- o
        """

        # Get Shape of the coupling array
        shape, _ = self.lat.coupling_shape(dx)

        # Return J for any given pair of sites u1 and u2
        p= model_params.get('p', 0.5)
        j = np.random.choice([-1, 1], size=shape, p=[p, 1 - p])

        return j

    # Override Internal Method to initialize the terms of the Hamiltonian
    def init_terms(self, model_params:dict) -> None:
        """
        Initialize the terms of the Hamiltonian
        """

        # We need to be careful here!!
        # This loops over the nearest neighbour directions (i.e. vertical & horiztonal)
        # -> This means each loop has to have an array such that each vertical or horizntal family member
        #    gets individual coupling strengths
        # -> This arrays needs to be of the same dimension as the actual lattice
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            curr_j = self.get_coupling_strength(model_params, u1, u2, dx)
            self.add_coupling(curr_j, u1, 'Sigmaz', u2, 'Sigmaz', dx)

