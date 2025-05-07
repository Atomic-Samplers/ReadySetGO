import matplotlib.pyplot as plt 
import numpy as np
from pathlib import Path

def energy_distribution_profile(atoms_list, directory='rsgo_results'):
    
    energy_array=np.array([e.get_potential_energy() for e in atoms_list])
    e_min=np.min(energy_array)
    energy_array=np.sort(energy_array)[::-1]-e_min
    
    
    plt.clf()
    plt.xlabel('Structures Ordered Energetically')
    plt.ylabel('Relative Energy to Lowest Energy Structure (eV)')
    plt.plot(energy_array, linestyle='-', marker='o')
     
    plt.savefig(Path(directory, 'energy_distribution_profile.png'))