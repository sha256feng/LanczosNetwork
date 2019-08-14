from deepchem.utils.save import load_sdf_files
import pickle

qm_tasks = [
    "property_0", "property_1", "property_2", "property_3", "property_4",
    "property_5", "property_6", "property_7", "property_8", "property_9",
    "property_10", "property_11"
]
dataframe = load_sdf_files(['ac19/10.sdf'], clean_mols=False, tasks=qm_tasks)
pickle.dump(dataframe, open('load_sdf_file.pickle', 'wb'))
