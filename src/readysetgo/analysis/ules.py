from re import L
import matplotlib.pyplot as plt 
import numpy as np
from pathlib import Path
import pandas as pd

def ules_plot(atoms_list, clustering_object, directory='rsgo_results', energy_range=0.025, global_minimum_energy=None ):
    energy_array=np.array([e.get_potential_energy() for e in atoms_list])
    if global_minimum_energy is None:
        global_minimum_energy = np.min(energy_array)

    dist_mat= clustering_object.get_dist_mat()

    def trim_distance_matrix(dist_mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Trims the distance matrix to only include rows and columns corresponding to the mask.
        """
        if dist_mat.size == 0:
            return dist_mat
        return dist_mat[mask][:, mask]

    def count_ules(iteration) -> int:
        trimmed_atoms_list= [atoms for atoms in atoms_list if atoms.info['id'] < iteration]
        trimmed_dist_mat = trim_distance_matrix(dist_mat, np.array([atoms.info['id'] < iteration for atoms in atoms_list]))

        energy_array = np.array([atoms.get_potential_energy()-global_minimum_energy for atoms in trimmed_atoms_list])
        low_energy_mask= np.array(energy_array) < global_minimum_energy + energy_range
        id_array=np.array([e.info['id'] for e in trimmed_atoms_list])
        ids_to_cluster= id_array[low_energy_mask]
        low_energy_trimmed_atoms_list = [atoms for atoms in trimmed_atoms_list if atoms.info['id'] in ids_to_cluster]
        low_energy_trimmed_dist_mat = trim_distance_matrix(trimmed_dist_mat, low_energy_mask)

        clustering_object.set_attribute('atoms_list', low_energy_trimmed_atoms_list)
        clustering_object.set_attribute('dist_mat', low_energy_trimmed_dist_mat)


        group=clustering_object.group()
        return len(group)

    ules_dict = {}
    for i in range(len(atoms_list)):
        ules_dict[i] = count_ules(i)
    
    df = pd.DataFrame({'Iteration':ules_dict.keys(), 'ULES Count':ules_dict.values()})
    df.to_csv(Path(directory, 'ules.csv'), index=False)
    
    plt.clf()
    
    plt.plot(df['Iteration'], df['ULES Count'], linestyle='-')  # Line in background
    # sc = plt.scatter(df.index, df['Energy (eV)'], c=df['ID'], cmap='viridis')
    # plt.colorbar(sc, label='ID')

    plt.xlabel('ReadySetGO Itearation')
    plt.ylabel('Unique Lowe Energy Structures (ULES) Count')
     
    plt.savefig(Path(directory, 'ules.png'))

    return plt