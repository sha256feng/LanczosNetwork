import pickle
train, valid, test = pickle.load( open('ac19/dc_feat_datasets.p', 'rb'))

print("train", train)
print("shape = ", train.get_shape())
print("valid", valid)
#nshards = train.get_shard_size()
#print(nshards)
#for i in range(nshards):
#    print(train.get_shard(i))

print("test", test)
print("shape = ", test.get_shape())
print(test.get_shard(0))
