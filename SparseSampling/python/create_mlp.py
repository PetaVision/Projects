import os
import sys
sys.path.insert(0, os.path.abspath(os.environ['HOME']+'/Work/Libraries/caffe/python/')) # Caffe
import caffe
from caffe import layers as L, params as P, to_proto

lmdb_file_name = 'pvCifar_train_lmdb'
proto_file_name = 'mlp.prototxt'

#MLP params
batch_size = 64       #Batch size for training with SGD
num_ip1_params = 500  #Number of free parameters in ip1
ip1_lrm_weights = 1   #Learning rate multiplier for ip1 weights
ip1_lrm_bias = 2      #Learning rate multiplyer for ip1 bias
ip2_lrm_weights = 1   #Learning rate multiplier for ip2 weights
ip2_lrm_bias = 2      #Learning rate multiplier for ip2 bias

netSpec = caffe.NetSpec()
netSpec.data, netSpec.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb_file_name, \
        transform_param=dict(scale=1./255), phase=0, ntop=2)


netSpec.ip1 = L.InnerProduct(netSpec.data, num_output=num_ip1_params, \
        weight_filler={'type': 'xavier'}, \
        bias_filler={'type': 'constant', 'value': 0})
netSpec.ip1.fn.params['param'] = [{'lr_mult': ip1_lrm_weights, 'decay_mult': 1}, \
                                  {'lr_mult': ip1_lrm_bias, 'decay_mult': 1}]

netSpec.relu1 = L.ReLU(netSpec.ip1, in_place=True)

netSpec.ip2 = L.InnerProduct(netSpec.data, num_output=10, \
        weight_filler={'type': 'xavier'}, \
        bias_filler={'type': 'constant', 'value': 0})
netSpec.ip2.fn.params['param'] = [{'lr_mult': ip2_lrm_weights, 'decay_mult': 1}, \
                                  {'lr_mult': ip2_lrm_bias, 'decay_mult': 1}]

netSpec.accuracy = L.Accuracy(netSpec.ip2, netSpec.label, phase=0)

netSpec.loss = L.SoftmaxWithLoss(netSpec.ip2, netSpec.label)

open(proto_file_name,"w").write(str(netSpec.to_proto()))
