import h5py
import numpy as np 


f1 = h5py.File("datasets/fer2013/hdf5/test.hdf5", 'r')
for key in f1.keys():
    print(key)

list(f1.keys())
X1 = f1['images']
y1 = f1['labels']
df1 = np.array(X1.value)
dfy1 = np.array(y1.value)
print(df1.shape)
print(dfy1.shape)
