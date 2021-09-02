# Efficient Discovery of Azoarene Photoswitches with Active Search and Quantumn Chemical Calculations

(Under review)

## Requirements

Gaussian16

Access to a High Performance Computing Cluster

Matlab: [https://www.mathworks.com/products/matlab.html](https://www.mathworks.com/products/matlab.html)

Python 3: [https://www.python.org/](https://www.python.org/)
- Further required libraries are included in `requirements.txt`.

## Data generation

In the `data_set` subfolder, the repository contains the code used to generate the 255,991 photoswitch dataset (pymolgen.py) along with the text file of the generated molecules. It also contains a csv file of the initial 198 molecules that were used to train the AS (`AS-training-data.csv`).

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

## Quantumn calculations

Quantum chemical calculations were performed using PyFlow [https://github.com/kuriba/PyFlow]

## Visualization
The `figures` subfolder contains the code used to generate Figure 11. The data used is available within the csv file UMAP-AllHits.csv. The `UMAP-proj.py` script generates UMAP plots from Morgan fingerprints and λmax values. It loads in the csv data, and converts the fingerprint data of each element into a numpy array of integers. The arrays of fingerprint data are stacked and then projected to 2d using the umap-learn package: https://umap-learn.readthedocs.io/en/latest/. These projections are visualized on a scatterplot, where they are colored by their corresponding vertical excitation energy values. Values above 700 or below 400 will be displayed in grey. In the case of our data, only values below 700 are present. The `box_plots.py` script generates a figure with a box plot of vertical excitation energy values for each Core ID discussed in the paper. Please refer to the manuscript for the definition of a Core ID.
