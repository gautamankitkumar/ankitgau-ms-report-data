import os

import torch
from ase.db import connect

from utils.fp_calculator import set_sym, db_to_fp
from utils.train_agent import Agent, get_scaling

# set symm func parameters
elements = ['Cu', 'Ag', 'Au']

Gs = [2]
cutoff = 6.0
g2_etas = [0.05, 4.0, 20.0, 80.0]
g2_Rses = [0.0]
# g4_etas = [0.005]
# g4_zetas = [1.0, 4.0]
# g4_lambdas = [-1.0, 1.0]

# params_set = set_sym(elements, Gs, cutoff, g2_etas=g2_etas, g2_Rses=g2_Rses, g4_etas=g4_etas, g4_zetas=g4_zetas, g4_lambdas=g4_lambdas)
params_set = set_sym(elements, Gs, cutoff, g2_etas=g2_etas, g2_Rses=g2_Rses)

# work folder
Name = 'CuAgAu'
if not os.path.isdir(f'./{Name}'):
    os.mkdir(f'./{Name}')

# calculate fingerprints for databases
# train_db = connect('./datasets/train.db')
# train_data = db_to_fp(train_db, params_set)
# torch.save(train_data, f'./{Name}/CuAgAu-train-dft.sav')

# valid_db = connect('./datasets/train.db')
# valid_data = db_to_fp(valid_db, params_set)
# torch.save(valid_data, f'./{Name}/CuAgAu-valid-dft.sav')

# test_db = connect('./datasets/test.db')
# test_data = db_to_fp(test_db, params_set)
# torch.save(test_data, f'./{Name}/CuAgAu-test-dft.sav')


# load data
train_data = torch.load(f'./{Name}/CuAgAu-train-dft.sav')   
valid_data = torch.load(f'./{Name}/CuAgAu-valid-dft.sav')
test_data = torch.load(f'./{Name}/CuAgAu-test-dft.sav')
scale_file = f'./{Name}/scale.sav'

if not os.path.isfile(scale_file):
    scale = get_scaling(train_data)
    torch.save(scale, scale_file)
else:
    scale = torch.load(scale_file)

# scale training fp
train_data['b_fp'] = (train_data['b_fp'] - scale['fp_min']) / (scale['fp_max'] - scale['fp_min'])
valid_data['b_fp'] = (valid_data['b_fp'] - scale['fp_min']) / (scale['fp_max'] - scale['fp_min'])

device = torch.device('cpu')
# for key in train_data.keys():
# 	train_data[key] = train_data[key].to(device)
# 	valid_data[key] = valid_data[key].to(device)

layer_nodes = [10,10]
activations = ['tanh','tanh']
lr = 0.1

# create model and train
element = torch.tensor([29, 47, 79])  # should have the same order with the elements above
model_paths = [f'./{Name}/model_for_{i}.sav' for i in element.tolist()]
log_name = f'./{Name}/train_log.txt'

agent = Agent(train_data=train_data, valid_data=valid_data, model_paths=model_paths, test_data=test_data,
              layer_nodes=layer_nodes,
              activation=activations, lr=lr, max_iter=20, history_size=100, device=device)

agent.train(log_name=log_name, n_epoch=100, interupt=True, val_interval=1,
            is_force=False, nrg_convg=2, force_convg=20, nrg_coef=1, force_coef=1)
# Energy convergence in meV, Force convergence in meV/Angstrom  
# No Force fitting from the data
