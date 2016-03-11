import numpy as np
import lmdb
import caffe
import IPython

# CIFAR has 60000 training images
num_frames = -1 # [int] number of frames to load in (all if <0)

#input_file = 'lmdb_datasets/pvCifar_test_lmdb_v1_samples_11'
input_file = '/Users/dpaiton/Work/Datasets/CIFAR/cifar10-val/cifar10_val_lmdb'

env = lmdb.open(input_file, readonly=True)
datum = caffe.proto.caffe_pb2.Datum()

x = []
y = []

with env.begin() as txn:
    for key, value in txn.cursor():
        if int(key) == num_frames:
            break
        datum.ParseFromString(value)
        flat_x = np.array(datum.float_data)
        if flat_x.size == 0:
            x.append(np.fromstring(datum.data, dtype=np.uint8).reshape(datum.channels, datum.height, datum.width))
            y.append(datum.label)
        else:
            x.append(flat_x.reshape(datum.channels, datum.height, datum.width))
            y.append(datum.label)
        print 'Loaded frame ' + key

x_array = np.vstack([x[i][np.newaxis,:] for i in range(len(x))])
y_array = np.array(y) # column array

IPython.embed()
