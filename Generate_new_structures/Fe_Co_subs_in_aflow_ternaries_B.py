# %%
import pandas as pd
from pymatgen.core import Structure, Element, Composition
from tqdm.auto import tqdm

# %%
df_B_new_structures=pd.read_json('../structure_ternaries_aflow/aflow_ternaries_processed_from_batches.json')

# %%
df_B_new_structures['structures']=[Structure.from_str(x,fmt='poscar') for x in df_B_new_structures['structures']]

# %%
progress_bar = tqdm(range(len(df_B_new_structures)))
for idx,row in df_B_new_structures.iterrows():
    
    B_mapping={}
    B_mapping_comp={}
    N_mapping={}
    N_mapping_comp={}
    first_transition=True
    for elem in row['structures'].elements:
        if elem.is_transition_metal and first_transition:
            B_mapping[elem]=Element("Fe")
            B_mapping_comp[elem]=Element("Co")
            N_mapping[elem]=Element("Fe")
            N_mapping_comp[elem]=Element("Co")
            first_transition=False
            
        elif elem.is_transition_metal and not first_transition:
            B_mapping[elem]=Element("Co")
            B_mapping_comp[elem]=Element("Fe")
            N_mapping[elem]=Element("Co")
            N_mapping_comp[elem]=Element("Fe")
        else:
            B_mapping[elem]=Element("B")
            B_mapping_comp[elem]=Element("B")
            N_mapping[elem]=Element("N")
            N_mapping_comp[elem]=Element("N")

    df_B_new_structures.loc[idx,'structures'].replace_species(species_mapping=B_mapping)

    progress_bar.update(1)  


# %%
df_B_new_structures['structures']=[x.as_dict() for x in df_B_new_structures['structures']]

# %%
df_B_new_structures.to_json('../structure_ternaries_aflow/B_new_structures.json')

print('process completed')