from rdkit import Chem
from rdkit.Chem import AllChem


def compute_morgan_features(
    smiles_list,
    feature_file_path,
    smiles_file_path=None,
    radius=2,
    nbits=2048,
    print_per=10,
    print_invalid=True,
):
    """
    compute morgan fingerprints for a list of smiles strings
    :param smiles_list: a list of smiles strings
    :param feature_file_path: the computed features will be stored in this file in a sparse format
    :param smiles_file_path:  if not None, the smiles strings will be printed in this file
    :param radius:  radius for computing the fingerprint
    :param nbits:  number of bits
    :param print_per:  print the progress
    :param print_invalid:  if True, print the invalid smiles if any
    """

    n = len(smiles_list)
    count = 0
    invalid_idx = []
    if smiles_file_path:
        open(smiles_file_path, "w").close()  # erase the content of this file
    with open(feature_file_path, "w") as f:
        for i in range(n):
            if i % print_per == 0:
                print(i, "/", n, end="\n", flush=True)
            smile = smiles_list[i].strip()
            mol = Chem.MolFromSmiles(smile)  # get molecule
            if mol is None:
                invalid_idx.append(i)
                if print_invalid:
                    print("invalid:", i, smile)
                continue
            count += 1
            if smiles_file_path:
                with open(smiles_file_path, "a") as g:
                    g.write(smile + "\n")
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
            nonzero_idx = list(fp.GetOnBits())
            for _, idx in enumerate(nonzero_idx):
                f.write("%d %d\n" % (count, idx + 1))

    return invalid_idx
