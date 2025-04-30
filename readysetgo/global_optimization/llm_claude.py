
import anthropic
import os
import numpy as np
from .core import GlobalOptimizerCore
from ..utils.common_functions import flatten_upper_triangle


class ClaudeGO(GlobalOptimizerCore):
    def __init__(self, base_atoms, db_path,  similarity=False, sim_list=[], close_contacts=False, lo=True, pbc=True):
        super().__init__(base_atoms, db_path)
        self.similarity = similarity
        self.sim_list = sim_list
        self.close_contacts = close_contacts
        self.lo = lo
        self.pbc = pbc

    def llm_client():
        return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def ask_claude(self, prompt, client, model="claude-3-7-sonnet-20250219"):
        """Send a prompt to Claude and return the response."""
        message = client.messages.create(
            model=model,
            system='You are a global optimiser. You are given a set of coordinates and you need to provide a new set of lower energy coordinates that are unique.',
            max_tokens=len(self.atoms)*5,  # Adjust based on your needs
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text

    def make_atoms_object(self, llm_coords, prompt, repsonse):
        """
        Convert the LLM coordinates to an ASE Atoms object.
        """
        # Convert the list of coordinates to a numpy array
        llm_coords = np.array(eval(llm_coords))

        # Create the Atoms object
        atoms = self.atoms.copy()
        atoms.set_positions(llm_coords[:, 1:4])
        atoms.set_chemical_symbols(llm_coords[:, 0].astype(str))
        atoms.info['prompt'] = prompt
        atoms.info['response'] = repsonse

        return atoms
    
    def go_suggest(self, db, client):
        llm_coords = None

         # Get the last entry in the database
        total_entries = db.count()
        last_entry = db.get(id=total_entries)
        last_atoms = last_entry.toatoms()

        for i in range(5): # selt a max of 5 times to get a valid response
            # Generate a new prompt based on the last coordinates
            if self.close_contacts:
                conditional_prompt = f'The last structure had atoms too close together.'
            elif self.similarity:
                conditional_prompt = f'The last structure was too similar to the coordinates of the follwoing structure(s): {self.sim_list}.'
            elif self.lo:
                conditional_prompt = f'your previous guess when we perfomrmed a geometry optmisation had the following coordinates {last_atoms.positions} and had an energy value of {last_atoms.get_potential_energy()}.'
            else:
                conditional_prompt = f'coordinates {last_atoms.positions} had an energy value of {last_atoms.get_potential_energy()}.'

            if self.pbc:
                pbc_prompt = 'Ensure periodic boundary conditions are accounted for.'
            else:
                pbc_prompt = 'Do not account for periodic boundary conditions.'

            default_prompt = f'Provide a list of coordinates for a structure with a unit cell of {self.atoms.cell} including the following list of atoms: {self.atoms.get_chemical_symbols()}. {pbc_prompt} Provide the structure in the format [[Atom,x1,y1,z1],[Atom,x2,y2,z2],...]. For example [[C,0.1,0.2,0.3],[C,0.4,0.5,0.6]]. Provide no other text. '

            prompt=f'{conditional_prompt} {default_prompt}'
            response = self.ask_claude(prompt, client)
            try:
                llm_coords = np.array(eval(response))
                break
            except SyntaxError or TypeError:
                print('failed with following response:\n', response)
        
        atoms=self.make_atoms_object(llm_coords, prompt, response)
        
        return atoms
    
  
