import numpy as np
from ase.geometry import get_distances


class GlobalDescriptorDistanceMatrix:
    """Descriptor Matrix for structures."""

    def __init__(self, atoms_list, verbose=0):
        self.atoms_list = atoms_list
        self.verbose = verbose
        self.descriptor_name = None
        self.invert = None

    def print_out(self):
        """Prints out the details of the descriptor matrix creation"""
        print("Creating Descriptor Matrix")
        print("Number of structures:", len(self.structure_list))
        print("Verbose level:", self.verbose)
        print(f"Using {self.descriptor_name} as Descriptor")

    def make_char_vec(self, structure):
        """Returns the characteristic vector from a given ase atoms object"""
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_pre_calculated_char_vecs(self, structure_row_list):
         # Precompute characteristic vectors for each structure
        return [self.make_char_vec(structure_row_list[x]) for x in range(len(structure_row_list))]
        

    def abs_dist_mat_entry(self, new_entry, dist_mat, char_vec_elements, precalculated_char_vecs):
        """Adds a new entry to the distance matrix"""
         
        # Calculate the distance score between structures and populate the distance matrix
        if new_entry >= len(dist_mat):
            dist_mat = np.r_[dist_mat, [np.zeros(new_entry)]]
            dist_mat = np.c_[dist_mat, np.zeros(new_entry + 1)]

        for i in range(new_entry + 1):
            if i == new_entry:
                distance_score = 0.0
            elif i < new_entry:
                distance_score = (
                    np.sum(
                        np.abs(
                            precalculated_char_vecs[new_entry]
                            - precalculated_char_vecs[i]
                        )
                    )
                    / char_vec_elements
                )
            dist_mat[new_entry, i] = dist_mat[i, new_entry] = np.round(
                distance_score, decimals=6
            )
        return dist_mat

    def normalise_dist_mat(self, dist_mat, invert=False):
        """Normalises the distance matrix"""
        
        dist_mat = dist_mat / np.max(dist_mat)
        
        if invert:
            dist_mat = 1 - dist_mat
        
        return dist_mat

    def make_dist_mat(self, structure_row_list, verbose):
        """Returns a normalised distance matrix for the structures in the structure_row_list"""

        if verbose > 0:
            self.print_out()

        # get the number of characterisitc elements for this structure
        char_vec_elements = len(self.make_char_vec(structure_row_list[0].toatoms()))
        if verbose > 0:
            print(
                "There are",
                char_vec_elements,
                "characteristic elements for this structure",
            )
        
        precalculated_char_vecs = self.get_pre_calculated_char_vecs(structure_row_list)

        file_num = len(structure_row_list)
        dist_mat = np.zeros((file_num, file_num))
        
        for j in range(file_num):
            dist_mat = self.abs_dist_mat_entry(new_entry=j, 
                                               dist_mat=dist_mat, 
                                               char_vec_elements=char_vec_elements, 
                                               precalculated_char_vecs=precalculated_char_vecs,
                                               )
        
        dist_mat=self.normalise_dist_mat(dist_mat, self.invert)
        
        return dist_mat
    

class AtomicDistancesDistanceMatrix(GlobalDescriptorDistanceMatrix):
    """Descriptor for distance matrix."""

    def __init__(self, atoms_list, verbose = 0):
        super().__init__(atoms_list, verbose)
        self.descriptor_name = "Atomic Distances"
        self.invert = False

    def make_char_vec(self, structure):
        """Returns the characteristic distance vector from a given ase atoms object"""
        dist_mat = np.array(get_distances(structure.positions, cell=structure.cell, pbc=structure.pbc)[1])
        upper = np.triu(dist_mat)
        flat = upper.flatten()
        no_zeroes = flat[flat != 0]
        char_vec = np.sort(no_zeroes)

        return char_vec
    
    def make_dist_mat(self, structure_row_list, verbose):
        return super().make_dist_mat(structure_row_list, verbose)

class InverseAtomicDistancesDistanceMatrix(GlobalDescriptorDistanceMatrix):
    """
    Descriptor for inverse distance matrix. Exhibit slightly better seperation in some casese
    than the non-inverse. Credit to Maximillian Ach for the suggestion.
    """
    def __init__(self, atoms_list, verbose = 0):
        super().__init__(atoms_list, verbose)
        self.descriptor_name = "Inverse Atomic Distances"
        self.invert = False

    def make_char_vec(self, structure):
        """Returns the characteristic distance vector from a given ase atoms object"""
        dist_mat = np.array(get_distances(structure.positions, cell=structure.cell, pbc=structure.pbc)[1])
        upper = np.triu(dist_mat)
        flat = upper.flatten()
        no_zeroes = flat[flat != 0]
        char_vec = np.sort(1 / no_zeroes)

        return char_vec

    def make_dist_mat(self, structure_row_list, verbose):
        return super().make_dist_mat(structure_row_list, verbose)