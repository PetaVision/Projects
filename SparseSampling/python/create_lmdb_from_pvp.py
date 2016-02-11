import os
import sys
import numpy as np
import scipy.sparse as sparse
import lmdb

# Add paths
sys.path.insert(0, os.path.abspath('../../../python/')) # PetaVision
sys.path.insert(0, os.path.abspath(os.environ['HOME']+'/Work/Libraries/caffe/python/')) # Caffe

# Import aux libraries
from pvtools import *
import caffe

#set up path to pvp file
pvp_file_path = os.environ['HOME']+'/Work/LANL/Data/S1_400.pvp'
image_list_path = os.environ['HOME']+'/Work/LANL/Data/mixed_cifar.txt'
lmdb_file_name = 'pvCifar_train_lmdb'

def pvObj2DenseMat(pvObj):
    numIm = len(pvObj)
    numF = pvObj.header['nf']
    numY = pvObj.header['ny']
    numX = pvObj.header['nx']
    dense_shape = numF * numY * numX
    out_mat = np.zeros((numIm, numF, numY, numX))
    for frame_idx in range(numIm):
        frame_indices = np.array(pvObj[frame_idx].values)[:,0].astype(np.int32)
        frame_data = np.array(pvObj[frame_idx].values)[:,1].astype(np.float32)
        numActive = len(frame_indices)
        ij_mat = (np.zeros(numActive), frame_indices)
        out_vec = np.array(sparse.coo_matrix((frame_data, ij_mat), shape=(1, dense_shape)).todense())
        out_mat[frame_idx,:,:,:] = out_vec.reshape((numF, numY, numX))
    return out_mat

def cifarList2Vec(file_loc,label_pos):
    label_list = []
    with open(file_loc, 'r') as f:
        for line in f:
            line_arry = line.split('/')
            label_list.append(line_arry[label_pos])
    return np.array(label_list)

pvData = readpvpfile(pvp_file_path) # This takes some time...
pvActivities = pvObj2DenseMat(pvData)
labels = cifarList2Vec(image_list_path, 6).astype(np.int64)
assert labels.shape[0] == pvActivities.shape[0]

# Database size set to be 10x bigger than needed
# as suggested in http://deepdish.io/2015/04/28/creating-lmdb-in-python/
map_size = pvActivities.nbytes * 10

env = lmdb.open(lmdb_file_name, map_size=map_size)
with env.begin(write=True) as txn:   # txn is a Transaction object
    for i in range(len(pvData)):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = pvActivities.shape[1]
        datum.height = pvActivities.shape[2]
        datum.width = pvActivities.shape[3]
        datum.data = pvActivities[i].tobytes()
        datum.label = int(labels[i])
        str_id = '{:08}'.format(i)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
