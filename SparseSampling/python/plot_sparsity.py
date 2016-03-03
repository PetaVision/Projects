import numpy as np
import re
import lmdb
import caffe
import matplotlib.pyplot as plt 
import IPython

version = 1 
num_perturbations = 10
log_files = ['pvCifar_test_lmdb_v'+str(version)+'_samples_'+str(x) for x in range(1,num_perturbations+1)]

total_frames = -1 # [int] number of frames to load in (all if <0)

l1_sparseness = []
l0_sparseness = []
for log_file in log_files:
   env = lmdb.open(log_file, readonly=True)
   datum = caffe.proto.caffe_pb2.Datum()

   x = []
   y = []
   with env.begin() as txn:
       for key, value in txn.cursor():
           if int(key) == total_frames:
               break
           datum.ParseFromString(value)
           flat_x = np.array(datum.float_data)
           x.append(flat_x.reshape(datum.channels, datum.height, datum.width))
           y.append(datum.label)
           print 'Loaded frame ' + key 

   x_array = np.vstack([x[i][np.newaxis,:] for i in range(len(x))])
   y_array = np.array(y) # column array
   
   l1_sparseness.append(np.sum(np.abs(x_array)))
   l0_sparseness.append(np.count_nonzero(x_array))

IPython.embed()
