# %%
import json
from json import JSONDecodeError
from urllib.request import urlopen

import pandas as pd
import json
from pymatgen.core import Composition
from pymatgen.core import Structure,Element

from collections import defaultdict
from tqdm.auto import tqdm
import pandas as pd 
import time
import os

import argparse

parser=argparse.ArgumentParser()
parser.add_argument("-b","--batch",type=int,help="start from 0 and go up to inclusive 14.")
args=parser.parse_args()

# %%
SERVER="https://aflow.org"
API="/API/aflux/?"

# Download a AFLUX response and return it as list of dictionaries
def aflux_request(matchbook, paging=1,entries=64, no_directives=False):
    request_url = SERVER + API + matchbook
    if not no_directives:
        request_url += f",$paging({paging},{entries}),format(json)"
    server_response = urlopen(request_url,)
    response_content  = server_response.read().decode("utf-8")
    # Basic error handling
    if server_response.getcode() == 200:
        try:
            return json.loads(response_content)
        except JSONDecodeError:
            pass
    print("AFLUX request failed!")
    print(f"  URL: {request_url}")
    print(f"  Response: {response_content}")
    return []

# %%
TM_list='Sc:Ti:V:Cr:Mn:Fe:Co:Ni:Cu:Zn:Y:Zr:Nb:Mo:Tc:Ru:Rh:Pd:Ag:Cd:Hf:Ta:W:Re:Os:Ir:Pt:Au:Hg'.split(":")
TM_sublist=['(Sc:Ti:V:Cr:Mn:Fe)','(Co:Ni:Cu:Zn:Y:Zr)','(Nb:Mo:Tc:Ru:Rh:Pd)','(Ag:Cd:Hf:Ta:W:Re)','(Os:Ir:Pt:Au:Hg)']
boron_group=['B','Al','Ga','In','Tl']
carbon_group=['C','Si','Ge','Sn','Pb']
pnictogens=['N','P','As','Sb','Bi']
chalcogens=['O','S','Se','Te']

# %%
batch=args.batch
step=2
batch_start=batch*step
batch_end=(batch+1)*step

for elem in TM_list[batch_start:batch_end]:
    
    for group_type in ['BoronGroup','CarbonGroup','Pnictogens','Chalcogens']:
        struct_ternary_aflow=defaultdict(list)
        
        for idx,TM_subgroup in enumerate(TM_sublist):
            more_data=True
            paging=1
            while more_data:
                
                try:
                    data = aflux_request(f"species({elem},{TM_subgroup},{group_type}),$nspecies(3)",paging=paging,entries=100)

                except TimeoutError:
                    print("Connection refused by the server..",flush=True)
                    time.sleep(60)
                    print("Was a nice sleep, now let me continue...",flush=True)
                    data = aflux_request(f"species({elem},{TM_subgroup},{group_type}),$nspecies(3)",paging=paging,entries=100)
                    
                if not data:
                    more_data=False
                    
                for record in data:
                    aurl=record['aurl']
                    REST_API='http://'+aurl.replace(':AFLOWDATA','/AFLOWDATA')+'/'
                    REQUEST =REST_API+"CONTCAR.relax.vasp"
                    try:
                        response =urlopen(REQUEST).read().decode('utf-8')
                        struct_ternary_aflow['structures'].append(response)
                        struct_ternary_aflow['auid'].append(record['auid'])
                    except:
                        continue
                
                print(f"Element: {elem} ,Group_type: {group_type} ,TM_sublist_idx:{idx}, Page:{paging},",flush=True)
                paging+=1

        if not os.path.exists(f'./aflow_ternaries_{elem}_{group_type}'):
            os.makedirs(f'./aflow_ternaries_{elem}_{group_type}')

        with open(f'./aflow_ternaries_{elem}_{group_type}/{elem}_{group_type}_ternary_structs.json','w') as outfile:
            json.dump(struct_ternary_aflow,outfile)



