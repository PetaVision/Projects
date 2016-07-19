import os
import sys
import re
sys.path.insert(0, os.path.abspath(os.environ['HOME']+'/Work/Libraries/caffe/python/')) # Caffe
import caffe
from caffe import layers as L, params as P, to_proto
from google.protobuf import text_format

lmdb_train_file = 'pvCifar_train_lmdb'
lmdb_test_file = 'pvCifar_val_lmdb'
proto_file = 'mlp.prototxt'
image_dataset = '/Users/dpaiton/Work/Datasets/CIFAR/cifar10/cifar10_train_lmdb/'

#MLP params
batch_size = 100      # Batch size for training with SGD
num_ip1_params = 768  # Number of free parameters in ip1
ip1_lrm_weights = 1   # Learning rate multiplier for ip1 weights
ip1_lrm_bias = 1      # Learning rate multiplyer for ip1 bias
ip2_lrm_weights = 1   # Learning rate multiplier for ip2 weights
ip2_lrm_bias = 1      # Learning rate multiplier for ip2 bias

train_data = caffe.NetSpec()
train_data.data, train_data.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb_train_file, phase=0, ntop=2)

test_data = caffe.NetSpec()
test_data.data, test_data.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb_test_file, phase=1, ntop=2)

merged_data = text_format.Merge(str(test_data.to_proto()), train_data.to_proto())

net = caffe.NetSpec()
net.data = L.Layer()
net.label= L.Layer()
net.ip1 = L.InnerProduct(net.data, num_output=num_ip1_params, \
        weight_filler={'type': 'xavier'}, \
        bias_filler={'type': 'constant', 'value': 0})
net.ip1.fn.params['param'] = [{'lr_mult': ip1_lrm_weights, 'decay_mult': 1}, \
                                  {'lr_mult': ip1_lrm_bias, 'decay_mult': 1}]
net.relu1 = L.ReLU(net.ip1, in_place=True)
net.ip2 = L.InnerProduct(net.relu1, num_output=10, \
        weight_filler={'type': 'xavier'}, \
        bias_filler={'type': 'constant', 'value': 0})
net.ip2.fn.params['param'] = [{'lr_mult': ip2_lrm_weights, 'decay_mult': 1}, \
                                  {'lr_mult': ip2_lrm_bias, 'decay_mult': 1}]
net.train_accuracy = L.Accuracy(net.ip2, net.label, phase=0)
net.test_accuracy = L.Accuracy(net.ip2, net.label, phase=1)
net.loss = L.SoftmaxWithLoss(net.ip2, net.label)

# net proto has 2 dummy layers, which occupy the first 112 chars
merged_net = text_format.Merge(str(net.to_proto())[112:], merged_data)

out_str = re.sub('phase: ([A-Z]{4,5})', 'include {\n    phase: \\1\n  }', str(merged_net))
open(proto_file,'w').write(str(out_str))
