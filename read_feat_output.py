import pickle

test_data = pickle.load(open('data/QM8/preprocess/QM8_preprocess_test_0000000.p', 'rb'))
#print(test_data)
print(test_data.keys())

#print("Node features")
#print(test_data['node_feat'])

#print("L multi")
#print(test_data['L_multi'])

#print('V_simple')
#print(test_data['V_simple'])

#print('label')
#print(test_data['label'])

#print('label weight')
#print(test_data['label_weight'])

train, valid, test = pickle.load( open('ac19/dc_feat_datasets.p', 'rb'))

print("train", train)

print("valid", valid)

print("test", test)
