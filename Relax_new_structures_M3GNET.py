# This file uses pre-trained M3GNet model to perform structural relaxations
# Author of M3GNet model: Tsz Wai Ko (Kenko)

import os
import warnings
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sqlitedict import SqliteDict

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import sys
from matgl.ext.ase import Relaxer

### Loading the pre-trained M3GNet PES model, which is trained on the MP-2021.2.8 dataset
import matgl

pot=matgl.load_model("/home/huseyin.ucar/matgl/pretrained_models/M3GNet-MP-2021.2.8-PES")
time.sleep(np.random.rand(1)[0]*10)

# device = "cpu"
# pot = pot.to(device)

# ### Do the following lines seperately before running this file:
# !rm files.txt
# !ls structures_lock > files.txt

with open("files.txt","r") as f:
    s = f.read()
s = s.split()
s = set(s)
h = {}  # using hashtag
for si in s:
    h[si] = None

    
with SqliteDict('/home/huseyin.ucar/relaxed_sqlites/B_new_structures_postSM.sqlite', autocommit=True) as db:
    length = len(db)
    # Define the new column name and the data to populate it
    final_struct_column = 'Final_Struct'
    final_energy_column = 'Final_Energy'
    initial_energy_column = 'Initial_Energy'

    for i in range(length):
        
        if  str(i)+'.lock' in h:
            continue
        
        if os.path.exists('./structures_lock/'+str(i)+'.lock'):
            continue
        f = open('./structures_lock/'+str(i)+'.lock','w')
        f.write("")
        f.close()
        
        value = db[i]
        structure = value[1].structures
        struct = Structure.from_dict(structure)

        st = time.time()
        relaxer = Relaxer(potential=pot.to('cpu'))
        relax_results = relaxer.relax(struct, fmax=0.01)
        # extract results
        final_structure = relax_results["final_structure"]
        final_energy = relax_results["trajectory"].energies[-1]
        initial_energy = relax_results["trajectory"].energies[0]
    
        # Iterate over the database items and update each record with the new column
        # Ensure we have a corresponding score for the current record
        value[1][final_struct_column] = final_structure.as_dict()
        value[1][final_energy_column] = final_energy
        value[1][initial_energy_column] = initial_energy
        db[i] = value