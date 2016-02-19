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

import IPython

pvp_file = os.environ['HOME']+'/Work/LANL/Data/S1_0Perturbation_Train.pvp'
image_list = os.environ['HOME']+'/Work/LANL/Data/mixed_cifar.txt'
lmdb_file = 'pvCifar_train_lmdb'      # [str] output destination
progress_write = 100                  # [int] How often to write out progress (number of frames)
write_separate_labels = False         # [bool] If True, will write txt file that only contains labels
num_validation = 10000                # [int] Size of validation dataset, if 0 will not make one
validation_file = 'pvCifar_val_lmdb'  # [str] Name of pvp validation file if num_validaiton > 0

assert(num_validation >= 0)

def pvObj2DenseMat(pvObj, progress_write):
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
        if not i%progress_write:
            print "Converted frame " + str(frame_idx) + " of " + str(numIm)
    return out_mat.astype('float')

def cifarList2Vec(file_loc,label_pos):
    label_list = []
    with open(file_loc, 'r') as f:
        for line in f:
            line_arry = line.split('/')
            label_list.append(line_arry[label_pos])
    return np.array(label_list)

def write_lmdb(filename, map_size, progress_write, data, index_list):
    with lmdb.open(filename, map_size=map_size) as env:
        with env.begin(write=True) as txn:   # txn is a Transaction object
            for i in index_list:
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = data.shape[1]
                datum.height = data.shape[2]
                datum.width = data.shape[3]
                #logActivities = np.copy(data[i])
                #logActivities[np.nonzero(logActivities)] = np.log(logActivities[np.nonzero(logActivities)])
                datum = caffe.io.array_to_datum(data[i], int(labels[i]))
                keystr = '{:04}'.format(i)
                txn.put(keystr.encode('ascii'), datum.SerializeToString())
                if not i%progress_write:
                    print "Wrote frame "+str(i)+" of "+str(index_list[-1])

pvData = readpvpfile(pvp_file, progress_write) # This takes some time...
pvActivities = pvObj2DenseMat(pvData)
labels = cifarList2Vec(image_list, 6).astype(np.int64)

assert(labels.shape[0] == pvActivities.shape[0])

if write_separate_labels:
    label_out_text = open('labels.txt','w')
    for label in labels:
        label_out_text.write(str(label)+'\n')
    label_out_text.close()

# Database size set to be 10x bigger than needed
# as suggested in http://deepdish.io/2015/04/28/creating-lmdb-in-python/
map_size = pvActivities[0:len(pvData)-num_validation].nbytes * 10

# Create initial set
write_lmdb(lmdb_file, map_size, progress_write, pvActivities, range(len(pvData) - num_validation))

# Create validation set if required
if (num_validation > 0):
    print "-------\nWriting validation set\n-------"
    map_size = pvActivities[len(pvData)-num_validation:len(pvData)].nbytes * 10
    write_lmdb(validation_file, map_size, progress_write, pvActivities, range(len(pvData)-num_validation, len(pvData)))
