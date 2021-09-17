# Efficient Discovery of Azoarene Photoswitches with Active Search and Quantumn Chemical Calculations

(Under review)

## Usage 
This code can be implemented to iteratively search a dataset. The code presented in this repository was used to generate the dataset and apply active search to select molecules within a certain criteria. 

### Data generation

In the `data_set` subfolder, the repository contains the code used to generate the 255,991 photoswitch dataset (pymolgen.py) along with the text file of the generated molecules. The pymolgen script can be used to create pdb files or SMILES strings. The molecules are created in a combinatorial fashion, please refer to the manuscript for a detailed description. It also contains a csv file of the initial 198 molecules that were used to train the AS (`AS-training-data.csv`).

The pymolgen.py script requires a SMILES string and directory to write the output. See below for an example. 

```python pymolgen.py dir1 [Y]C1=C(C2=C3C4=CC=C2C=C1)C=CC3=C5C=CC6=C(C=CC7=CC=C4C5=C67)[Y]```

### Quantumn calculations

Quantum chemical calculations were performed using PyFlow [https://github.com/kuriba/PyFlow]. PyFlow was used to create a workflow that performs five subsequent calculations. Please refer to the manuscript for a detailed description. 

The workflow was set up using a json file and by actvatng the PyFlow environment. 
```python 
conda activate pyflow 

pyflow setup _workflow_name_ --config_file /path/to/config
```
Once the workflow was created, the `pymolgen.py` script was used to genrated a set of pdb files based on the molecules selected for each active search iteration. Those pdb files, which contained 4 conformers per molecule, were moved into the /unopt_pdbs folder of the workflow. The workflow was then submitted with `pyflow begin` 

## Requirements

Gaussian16

Access to a High Performance Computing Cluster

Matlab: [https://www.mathworks.com/products/matlab.html](https://www.mathworks.com/products/matlab.html)

Python 3: [https://www.python.org/](https://www.python.org/)
- Further required libraries are included in `requirements.txt`.

## Active search

The code repository contains the k-NN predictive model and our implementation of the active search algorithm in the active_search subfolder. In the `active_search `folder, subdirectories `active_learning`, `active_search`, and `efficient_nonmyopic_active_search` contain code from previous work in the active search literature, while subdirectory `process_molecules` contains code to process the molecules in our search space, specifically to compute the Morgan fingerprint (`utils.py`) and Tanimoto similarities (`calculate_nearest_neigihbors` files).

`choose_one_batch.m` implements the compilation of a batch of queries.
A typical printed output from this file follows:
```
> choose_one_batch
loading data...

problem =

  struct with fields:

               points: [255992x1 double]
           num_points: 255992
          num_classes: 2
           batch_size: 50
              verbose: 1
           do_pruning: 1
          num_initial: 285

final training data of size 723 723
computing the batch...

```

Refer to the README in `active_search` for a more detailed description.


## Visualization
The `figures` subfolder contains the code used to generate Figure 11. The data used is available within the csv file UMAP-AllHits.csv. The `UMAP-proj.py` script generates UMAP plots from Morgan fingerprints and λmax values. It loads in the csv data, and converts the fingerprint data of each element into a numpy array of integers. The arrays of fingerprint data are stacked and then projected to 2d using the umap-learn package: https://umap-learn.readthedocs.io/en/latest/. These projections are visualized on a scatterplot, where they are colored by their corresponding vertical excitation energy values. Values above 700 or below 400 will be displayed in grey. In the case of our data, only values below 700 are present. The `box_plots.py` script generates a figure with a box plot of vertical excitation energy values for each Core ID discussed in the paper. Please refer to the manuscript for the definition of a Core ID.
