# the file should be run from the repository root

from pprint import pprint

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# res = unpickle('cifar-10-batches-py/batches.meta')
res = unpickle('cifar-10-batches-py/data_batch_1')
pprint(res)
print('done')
