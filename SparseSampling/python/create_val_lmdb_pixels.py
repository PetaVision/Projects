import lmdb
#import numpy as np
#import caffe
#import IPython

# CIFAR has 50,000 training images
num_train = 40000
num_val = 10000

input_file = '/Users/dpaiton/Work/Datasets/CIFAR/cifar10/cifar10_train_lmdb'
output_path = '/Users/dpaiton/Work/Datasets/CIFAR/cifar10-val/'

input_env = lmdb.open(input_file, readonly=True)
map_size = input_env.info()['map_size']
train_env = lmdb.open(output_path+'cifar10_train_lmdb', map_size=map_size )
val_env = lmdb.open(output_path+'cifar10_val_lmdb', map_size=map_size )

#datum = caffe.proto.caffe_pb2.Datum()

train_idx = 0
val_idx = 0
with input_env.begin() as txn:
    for key, value in txn.cursor():
            if int(key) < num_train:
                keystr = '{:04}'.format(train_idx)
                with train_env.begin(write=True) as txn:   # txn is a Transaction object
                    txn.put(keystr.encode('ascii'), value)
                train_idx += 1
                print 'Placed training image '+keystr
            else:
                keystr = '{:04}'.format(val_idx)
                with val_env.begin(write=True) as txn:   # txn is a Transaction object
                    txn.put(keystr.encode('ascii'), value)
                val_idx += 1
                print 'Placed validation image '+keystr

input_env.close()
train_env.close()
val_env.close()
