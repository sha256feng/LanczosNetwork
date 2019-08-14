import os
import glob
import pickle
import numpy as np
import deepchem as dc

if __name__ == '__main__':
    # Get the sdf data ready, "all.sdf", "all.sdf.csv" two files in the same folder;
    dataset_file = 'ac19/10.sdf'

    # Get to know the tasks
    qm8_tasks = [
        "property_0", "property_1", "property_2", "property_3", "property_4",
        "property_5", "property_6", "property_7", "property_8", "property_9",
        "property_10", "property_11"
    ]

    if featurizer == 'MP':
        featurizer = dc.feat.WeaveFeaturizer(
            graph_distance=False, explicit_H=True)

    # Utilize deepchem's tool to load SDF file
    loader = dc.data.SDFLoader(
        tasks=qm8_tasks,
        smiles_field="smiles",
        mol_field="mol",
        featurizer=featurizer)

    dataset = loader.featurize(dataset_file)

    print('Featurization finished!')


    print("Pair features: ", dataset.get_pair_features())

    print("Node featuers: ", dataset.get_atom_features())

    print("There are ", dataset.get_num_atoms(), " atoms.")

    print("There are ", dataset.get_num_features(), "features. ")
