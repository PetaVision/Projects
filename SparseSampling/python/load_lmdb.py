import numpy as np
import lmdb
import caffe
import IPython

# CIFAR has 60000 training images
num_frames = -1 # [int] number of frames to load in (all if <0)

env = lmdb.open('lmdb_datasets/pvCifar_test_lmdb_v1_samples_11', readonly=True)
datum = caffe.proto.caffe_pb2.Datum()

x = []
y = []

with env.begin() as txn:
    for key, value in txn.cursor():
        if int(key) == num_frames:
            break
        datum.ParseFromString(value)
        flat_x = np.array(datum.float_data)
        x.append(flat_x.reshape(datum.channels, datum.height, datum.width))
        y.append(datum.label)
        print 'Loaded frame ' + key

x_array = np.vstack([x[i][np.newaxis,:] for i in range(len(x))])
y_array = np.array(y) # column array

IPython.embed()
