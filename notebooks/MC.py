import os
import random
from time import time

import numpy as np
import torch
from ase.build import fcc111
from ase.io import write
from ase.neighborlist import NeighborList
from ase.units import kB

from utils.fcc_helpers import conditional_cal_nrg_local, create_clusters
from utils.fp_calculator import set_sym
from utils.train_agent import BPNN

# load models and symmetry functions
Name = 'CuAgAu'
scale = torch.load(f'./{Name}/scale.sav')
layer_nodes = [10, 10]
activations = ['tanh', 'tanh']
elements_i = [29, 47, 79]
n_fp = 12
models = [BPNN(n_fp, layer_nodes, activations) for _ in elements_i]
model_paths = [f'./{Name}/model_for_{i}.sav' for i in elements_i]
for i in range(len(models)):
    models[i].load_state_dict(torch.load(model_paths[i], map_location=torch.device('cpu')))

# set symm func parameters
elements = ['Cu', 'Ag', 'Au']  # [1, 2, 3 in the param_set]
Gs = [2]
cutoff = 6.0
g2_etas = [0.05, 4.0, 20.0, 80.0]
g2_Rses = [0.0]
# g4_etas = [0.005]
# g4_zetas = [1.0, 4.0]
# g4_lambdas = [-1.0, 1.0]

# params_set = set_sym(elements, Gs, cutoff, g2_etas=g2_etas, g2_Rses=g2_Rses, g4_etas=g4_etas, g4_zetas=g4_zetas, g4_lambdas=g4_lambdas)
params_set = set_sym(elements, Gs, cutoff, g2_etas=g2_etas, g2_Rses=g2_Rses)

# create slab
n_Cu, n_Ag, n_Au, n_total = 500, 500, 500, 1500
lc_Cu, lc_Ag, lc_Au = 3.6387, 4.1628, 4.1733
c_Cu, c_Ag, c_Au = n_Cu / n_total, n_Ag / n_total, n_Au / n_total
lc = c_Cu * lc_Cu + c_Ag * lc_Ag + c_Au * lc_Au
slab = fcc111('Au', size=(10, 10, 15), vacuum=6.0, a=lc)
slab.set_pbc([1, 1, 0])
ids = np.arange(0, n_total).tolist()
random.shuffle(ids)
chem_symbols = ['Au'] * n_total  # used to create slab only
for i in ids[:n_Cu]:
    chem_symbols[i] = 'Cu'
for i in ids[n_Cu:n_Cu + n_Ag]:
    chem_symbols[i] = 'Ag'
slab.set_chemical_symbols(chem_symbols)
write('old_slab.traj', slab)

# make store directory
T = 800  # MC temperature
task_name = f'CuAgAu-{n_Cu}-{n_Ag}-{n_Au}-{T}'
subfolder = 'CuAgAu-MC'
if not os.path.isdir(f'{Name}/{subfolder}'):
    os.mkdir(f'{Name}/{subfolder}')

with open(f'{Name}/{subfolder}/{task_name}-initial-symbols.txt', 'w') as f:
    f.write(' '.join(chem_symbols))

logfile = open(f'{Name}/{subfolder}/{task_name}-MC-log.txt', 'w')
logfile.write(f'ind1 ind2 sym1 sym2 is_change \r\n')

# initialize MC
attempt, success = 0, 0
nl = NeighborList([cutoff] * len(slab), skin=0.01, bothways=True, self_interaction=False)
nl.update(slab)
Cu_bin = np.arange(0, n_Cu).tolist()
Ag_bin = np.arange(n_Cu, n_Cu + n_Ag).tolist()
Au_bin = np.arange(n_Cu + n_Ag, n_total).tolist()
dif_bins = {'Cu': Ag_bin + Au_bin, 'Ag': Cu_bin + Au_bin, 'Au': Cu_bin + Ag_bin}
symbols = ['Cu'] * n_Cu + ['Ag'] * n_Ag + ['Au'] * n_Au  # used in MC
n_step = 20000  # number of successful exchanges

t1 = time()
while success < n_step:
    # randomly select two different atoms to exchange
    pointer_1 = random.randint(0, n_total - 1)
    sym_1 = symbols[pointer_1]
    atom_id_1 = ids[pointer_1]  # atom id in slab
    pointer_2 = random.sample(dif_bins[sym_1], k=1)[0]
    sym_2 = symbols[pointer_2]
    atom_id_2 = ids[pointer_2]
    atom_ids = [atom_id_1, atom_id_2]

    cluster, conditions = create_clusters(slab, nl, atom_ids, cutoff)
    nrg1 = conditional_cal_nrg_local(models, cluster, params_set, elements, scale, conditions)

    # set up a temp slab
    temp_slab = slab.copy()
    temp_slab[atom_id_1].symbol, temp_slab[atom_id_2].symbol = sym_2, sym_1

    cluster, conditions = create_clusters(temp_slab, nl, atom_ids, cutoff)
    nrg2 = conditional_cal_nrg_local(models, cluster, params_set, elements, scale, conditions)

    d_nrg = nrg2 - nrg1

    is_swap = False
    # atomic swap
    if d_nrg < 0 or np.exp(-d_nrg / (kB * T)) > np.random.rand():
        slab = temp_slab
        is_swap = True
        success += 1
        ids[pointer_1], ids[pointer_2] = ids[pointer_2], ids[pointer_1]

    logfile.write(f'{atom_id_1} {atom_id_2} {sym_1} {sym_2} {is_swap} \r\n')
    attempt += 1

    if success % 2000 == 0 and success > 0:
        with open(f'{Name}/{subfolder}/{task_name}-t={success}-symbols.txt', 'w') as f:
            f.write(' '.join(slab.get_chemical_symbols()))

logfile.close()

t2 = time()
print(f'Code completed in {t2 - t1:0.4f} seconds')

write('new_slab.traj', slab)
