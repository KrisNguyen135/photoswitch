import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--iteration', type=int, default=1)  # last: 40
parser.add_argument('--policy', type=int, default=33)
args = parser.parse_args()

iteration = args.iteration
policy = args.policy
iter_dir = f'./data/iterations/iteration{iteration}'

smiles_file = './data/unlabeled.txt'
labels = np.loadtxt('./process_molecules/initial_labeled_data/labels', dtype=int)
num_labeled = labels.size

rec_dir = f'{iter_dir}/recommended_batch/'
chosen_ind_file = rec_dir + f'policy_{policy}_chosen_ind'
recommended_smiles_file = rec_dir + f'recommended_smiles_iteration{iteration}'
recommended_ind_file = rec_dir + f'recommended_ind_iteration{iteration}'

print(f'reading smiles file: {smiles_file}')
with open(smiles_file, 'r') as f:
    lines = f.readlines()
print(f'unlabeled pool size: {len(lines)}')

chosen_ind = np.loadtxt(chosen_ind_file, dtype=int)

python_chosen_ind = chosen_ind - num_labeled - 1
np.savetxt(recommended_ind_file, python_chosen_ind, fmt='%d')
print(f'recommended ind file path: {recommended_ind_file}')

with open(recommended_smiles_file, 'w') as f:
    for ind in python_chosen_ind:
        f.write(lines[ind])
print(f'recommended smiles file path: {recommended_smiles_file}')
