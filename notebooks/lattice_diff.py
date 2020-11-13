import os
import torch
import numpy as np
from ase.db import connect
from utils.fcc_helpers import cal_nrg
from utils.train_agent import BPNN
from utils.fp_calculator import set_sym
from ase.build import fcc111
import matplotlib.pyplot as plt

m = 1
n = 4000
np.random.seed(seed=42)
lc_Cu, lc_Ag, lc_Au = 3.6387, 4.1628, 4.1733

full_data = connect('./datasets/CuAgAu.db')
p =[]
for i in range(m,n+1):
    if i%200==0:
        print(i)
    atoms = full_data.get_atoms(selection=i)
    X = atoms.get_atomic_numbers().tolist()
    F, Cu, Ag, Au = atoms.get_volume(),X.count(29),X.count(47),X.count(79)
    lat = (F * 4/len(atoms))**(1/3)
    lat_ref = (Cu * lc_Cu + Ag*lc_Ag + Au*lc_Au)/len(atoms)
    plot_ar = [i,lat]
    p.append(plot_ar)
    # if i in test_ids:
    #     test_en.append([i,(atoms.get_potential_energy()-ans)/len(atoms)])
    # else:
    #     train_en.append([i,(atoms.get_potential_energy()-ans)/len(atoms)])


# plot it
train_en = np.array(p)
fig = plt.figure()
a0 = fig.add_subplot(111)

a0.plot(train_en[:,0],train_en[:,1],'b.')
a0.set_xlabel('Calculation ID')
a0.set_ylabel(r"Actual - Vegard's law $\AA$")
a0.set_title('Lattice constants difference')
a0.set_xlim([m,n])
a0.legend([f'Difference in Lattice constants'])
plt.show()
