import matplotlib.pyplot as plt 
import numpy as np
from pathlib import Path
import pandas as pd

def energy_distribution_profile(atoms_list, directory='rsgo_results'):
    
    energy_id_array=np.array([[e.get_potential_energy(), e.info['id']] for e in atoms_list])
    df = pd.DataFrame(energy_id_array, columns=['Energy (eV)', 'ID'])
    e_min=np.min(df['Energy (eV)'])
    df=df.sort_values(by='Energy (eV)', ascending=False, ignore_index=True)
    df['Energy (eV)'] = df['Energy (eV)'] - e_min
    
    plt.clf()
    
    plt.plot(df.index, df['Energy (eV)'], linestyle='-', color='gray', alpha=0.5)  # Line in background
    sc = plt.scatter(df.index, df['Energy (eV)'], c=df['ID'], cmap='viridis')
    plt.colorbar(sc, label='ID')

    plt.xlabel('Structures Ordered Energetically')
    plt.ylabel('Relative Energy to Lowest Energy Structure (eV)')
     
    plt.savefig(Path(directory, 'energy_distribution_profile.png'))