# %%
from __future__ import annotations

import os
import shutil
import warnings
import zipfile
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from dgl.data.utils import split_dataset
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn
from matgl.models import M3GNet
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelLightningModule


import numpy as np
import json

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

# %%
with open('../structure_magmom_dataset/mp_struct_magmom_ds.json','r') as f:
    struct_magmom_ds=json.load(f)

df=pd.DataFrame.from_dict(struct_magmom_ds)
df.loc[df['orderings']=='Unknown']

targets=[]
structures=[]

for struct_str,target in zip(struct_magmom_ds['structures'],struct_magmom_ds['total_mags_norm_vol']):
    struct=Structure.from_dict(struct_str)
    structures.append(struct)
    targets.append(target)

mean=np.mean(targets)
sdev=np.std(targets)


# %%
targets=[(el-mean)/sdev for el in targets]


# %%
# get element types in the dataset
elem_list = get_element_list(structures)
# setup a graph converter
converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
# convert the raw dataset into M3GNetDataset
mp_dataset = MGLDataset(
    threebody_cutoff=4.0,
    structures=structures,
    converter=converter,
    labels={"magmom": targets},
    include_line_graph=True,
)

# %%
train_data, val_data, test_data = split_dataset(
    mp_dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)
my_collate_fn = partial(collate_fn, include_line_graph=True)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=my_collate_fn,
    batch_size=128,
    num_workers=0,
)

# %%
# setup the architecture of M3GNet model
blank_model = M3GNet(
    is_intensive=True,
    readout_type="set2set",
)

# %%
model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")

# %%
blank_dict=blank_model.state_dict()

# %%
model.model.load_state_dict(blank_dict)

# %%
model.transformer.mean=mean
model.transformer.std=sdev

# %%
lit_module = ModelLightningModule(model=model.model,include_line_graph=True)


# %%
logger = CSVLogger("logs", name="magmom-from-scratch")
trainer = pl.Trainer(max_epochs=2000, accelerator="gpu", logger=logger)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


# %%
model_export_path = "./trained_model/"
if not os.path.exists(model_export_path):
    os.makedirs(model_export_path)
model.save(model_export_path)

# %%
# # This code just performs cleanup for this notebook.

# for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
#     try:
#         os.remove(fn)
#     except FileNotFoundError:
#         pass

# shutil.rmtree("logs")

# %%



