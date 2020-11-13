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

full_data = connect('./datasets/CuAgAu.db')

# if os.path.exists('./test.db'):
#     os.remove('./test.db')
# if os.path.exists('./train.db'):
#     os.remove('./train.db')
# if os.path.exists('./valid.db'):
#     os.remove('./valid.db')


train_data = connect('./datasets/train.db')
valid_data = connect('./datasets/valid.db')
test_data = connect('./datasets/test.db')

# # Generate different train and test dataset
valid_and_test_ids = np.random.choice(np.arange(1,n),n//5,replace=False)

valid_ids = valid_and_test_ids[:n//10]
test_ids = valid_and_test_ids[n//10:]


# for i in range(1,n+1):
#     if i%100==0:
#         print(i)
#     row = full_data.get_atoms(selection = i)
#     if i in test_ids:
#         test_data.write(row)
#     elif i in valid_ids:
#         valid_data.write(row)
#     else:
#         train_data.write(row)


# load models and symmetry functions
Name = 'CuAgAu'
scale = torch.load(f'./{Name}/scale.sav')
layer_nodes = [10,10]
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

params_set = set_sym(elements, Gs, cutoff, g2_etas=g2_etas, g2_Rses=g2_Rses)

train_en = []
test_en = []
valid_en = []
for i in range(m,n+1):
    if i%200==0:
        print(i)
    atoms = full_data.get_atoms(selection=i)
    ans = cal_nrg(models, atoms, params_set, elements, scale)
    X = atoms.get_atomic_numbers().tolist()
    # print(i,atoms.get_cell_lengths_and_angles(),X.count(29),X.count(47),X.count(79))
        
    if i in test_ids:
        test_en.append([i,(atoms.get_potential_energy()-ans)/len(atoms)])
    elif i in valid_ids:
        valid_en.append([i,(atoms.get_potential_energy()-ans)/len(atoms)])
    else:
        train_en.append([i,(atoms.get_potential_energy()-ans)/len(atoms)])


# plot it
train_en = np.array(train_en)
test_en = np.array(test_en)
valid_en = np.array(valid_en)
hist_en = np.concatenate((train_en[:,1],test_en[:,1],valid_en[:,1]))
f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})

a0.plot(train_en[:,0],train_en[:,1],'b.')
a0.set_xlabel('Calculation ID')
a0.set_ylabel('RMSE (eV/atom)')
a0.set_title('12-'+ '-'.join([str(x) for  x in layer_nodes]) + '-1 Framework')
a0.set_xlim([m,n])
a0.plot(valid_en[:,0],valid_en[:,1],'go')
a0.plot(test_en[:,0],test_en[:,1],'ro')
a0.axhline(0,color='k',ls='dashed')
a0.legend([f'Train Error {np.sqrt(np.mean((train_en[:,1])**2))*1000:.2f} meV/atom',
           f'Valid Error {np.sqrt(np.mean((valid_en[:,1])**2))*1000:.2f} meV/atom',
           f'Test Error {np.sqrt(np.mean(test_en[:,1]**2))*1000:.2f} meV/atom'])

a1.hist(hist_en,bins=20,density=True,orientation='horizontal')
a1.set_xlabel('Probability')
a1.set_title(fr'Error Distribution, \n $\mu$ = {np.mean(hist_en):.4f} eV/atom, $\sigma$ =  {np.std(hist_en):.4f} eV/atom')
plt.show()
