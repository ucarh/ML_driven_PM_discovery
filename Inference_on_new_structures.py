# %%
from sqlitedict import SqliteDict
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm

import matgl
import joblib
import time
import os

from pymatgen.core import Structure

# %%
time.sleep(np.random.rand(1)[0]*10)

# %%
import sys
sys.path.append('/home/huseyin.ucar/matminer/')

# %%
from sklearn.model_selection import StratifiedKFold,KFold


from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier



# %%
import  matminer.featurizers.structure as st
import matminer.featurizers.site  as site
from matminer.featurizers.base import MultipleFeaturizer


# %%
database_path = '/home/huseyin.ucar/pred_relaxed_sqlites/B_new_structures_postSM.sqlite'

# %%
magmom_model=matgl.load_model('/l/users/huseyin.ucar/megnet_magnets/25-M3GNET-magmom-from-scratch-NOFZ-Eform-Elm-Emb/trained_model/epoch_750')

# %%
MAE_RF_model=joblib.load("./MAE_RF.joblib")

# %%
Tc_RF_model=joblib.load("./Tc_RF.joblib")

# %%
featurizers=MultipleFeaturizer(
    [
        # st.SiteStatsFingerprint.from_preset('ucar_local_prop_diff'),  #1;  1+2=0.632
        # st.SiteStatsFingerprint.from_preset('ucar_elemental_prop'), #2
        # st.SiteStatsFingerprint(site.BondOrientationalParameter()), #3,  1+2+3=0.637
        #st.SiteStatsFingerprint(site.AverageBondLength(method=VoronoiNN(),)) #4 #1+2+4=0.637
        #st.SiteStatsFingerprint(site.AverageBondAngle(method=VoronoiNN(),)) #5, #1+2+4=0.633
        
        
        st.SiteStatsFingerprint(site.AGNIFingerprints()), #6  alone =0.653, 1+2+6=0.637..
        st.SiteStatsFingerprint(site.OPSiteFingerprint()), #7, along=0.643  6+7=0.661
        # st.SiteStatsFingerprint(site.VoronoiFingerprint()), #10 alone =0.66, 6+7+10=0.67, 1+2+6+7+10=0.65
        st.OrbitalFieldMatrix(), #18, 0.635 , 6+7+10+18=0.677 after dropping 0 values in columns with .any()



        # st.SiteStatsFingerprint(site.CrystalNNFingerprint.from_preset('ops')), #8 alone =0.64 , 6+7+8=0.655
        #st.SiteStatsFingerprint(site.CrystalNNFingerprint.from_preset('cn')), #9 alone =0.627
        # st.SiteStatsFingerprint(site.IntersticeDistribution()) #11, 0.642, 6+7+10+11=0.675
        #st.SiteStatsFingerprint(site.GaussianSymmFunc()), #12 alone= 0.626
        #st.BondFractions.from_preset('VoronoiNN'), #12, 0.622
        #st.BagofBonds(), #13 took very long
        # st.StructuralHeterogeneity(), #14, 0.627, 6+7+10+14=0.659
        #st.MinimumRelativeDistances(), #15, 0.614
        
        #st.CoulombMatrix() #16, 0.62
        #st.SineCoulombMatrix(), #17, 0.634
        #st.XRDPowderPattern(), #19, 0.638, 6+7+10+19=0.663
        # st.RadialDistributionFunction() #20, 0.643, 6+7+10+20=0.666


        # st.GlobalSymmetryFeatures(),
        # st.DensityFeatures(),
        # st.ChemicalOrdering(),    #ALL basic ones= 0.652
        # st.MaximumPackingEfficiency(),
        # st.StructuralComplexity()
    ]
)
with open("files.txt","r") as f:
    s = f.read()
s = s.split()
len(s)
s = set(s)
h = {}
for si in s:
    h[si] = None
# %%
with SqliteDict(database_path, autocommit=True) as db:
    length = len(db)
    for i in range(length):
        
        if  str(i)+'.lock' in h:
            continue
        
        if os.path.exists('./structures_lock/'+str(i)+'.lock'):
            continue
        
        f = open('./structures_lock/'+str(i)+'.lock','w')
        f.write("")
        f.close()
        
        value = db[i]
        
        structure = value[1].Final_Struct
        struct = Structure.from_dict(structure)

        try:
            pred_MAE=MAE_RF_model.predict(np.array(featurizers.featurize(struct)).reshape(1,-1))[0]
            pred_Tc=Tc_RF_model.predict(np.array(featurizers.featurize(struct)).reshape(1,-1))[0]
            pred_magmom=magmom_model.predict_structure(struct).numpy()
        
            value[1]['pred_MAE']=pred_MAE
            value[1]['pred_Tc']=pred_Tc
            value[1]['pred_magmom']=pred_magmom
            value[1]['formula']=struct.reduced_formula
            value[1]['volume']=struct.volume
            
        except Exception:
            value[1]['pred_MAE']=np.nan
            value[1]['pred_Tc']=np.nan
            value[1]['pred_magmom']=np.nan
            value[1]['formula']=np.nan
            value[1]['volume']=np.nan
        
        db[i] = value
        
        # df_dict['structures'].append(value[1].structures)
        # df_dict['auids'].append(value[1].auid)
        # df_dict['Final_structs'].append(value[1].Final_struct)
        # df_dict['Final_Energies'].append(value[1].Energy)
        


