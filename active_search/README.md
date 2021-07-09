# Active Search

This subdirectory contains active-search-related code.

## Toolboxes

Subdirectories `active_learning`, `active_search`, and `efficient_nonmyopic_active_search` contain code from previous work in the active search literature.

Our main search algorithm is implemented in `efficient_nonmyopic_active_search`.
Please refer to [https://github.com/shalijiang/efficient_nonmyopic_active_search](https://github.com/shalijiang/efficient_nonmyopic_active_search) for more details on how to use this code.

## Processing

Subdirectory `process_molecules` contains code to process the molecules in our search space, specifically to compute the Morgan fingerprint (`utils.py`) and Tanimoto similarities (`calculate_nearest_neigihbors` files).

## Workflow

- `choose_one_batch.m` implements the compilation of a batch of queries.
- `load_data.m` processes and reads in our dataset.
- `get_the_batch_of_smiles.py` generates the recommended SMILES strings.
- `core_selector` files implement the core-equidistribution requirement.
