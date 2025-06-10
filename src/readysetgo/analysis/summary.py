from readysetgo.analysis import energy_distribution_profile, ules
from matplotlib import pyplot as plt
import numpy as np


def plot_2xn_analysis(name_atoms_dict: dict, bins: int = 200)->None:
    rows=int(np.ceil(len(name_atoms_dict)/2))
    
    # fig, axes = plt.subplots(rows, 2, figsize=(10, 5*rows))

    # def plot_bond_lengths(ax, atoms_list, title):
    #     all_distances = []
    #     for atoms in atoms_list:
    #         atoms=atoms[np.array(atoms.get_chemical_symbols()) == 'C']
    #         distances = atoms.get_all_distances(mic=True)
    #         distances = np.triu(distances)
    #         distances = distances.flatten()
    #         distances = distances[distances > 0]
    #         all_distances.extend(distances)

    #     ax.hist(all_distances, bins=bins)
    #     ax.set_xlabel('C-C Distances (Angstrom)')
    #     ax.set_ylabel('Frequency')
    #     ax.set_title(title)

    # axes_count=0
    # for key, atoms_list in name_atoms_dict.items():
    #     row=axes_count//3
    #     column=axes_count%3
    #     plot_bond_lengths(axes[row, column], atoms_list, key)
    #     axes_count+=1
    

    # plt.tight_layout()
    # plt.show()