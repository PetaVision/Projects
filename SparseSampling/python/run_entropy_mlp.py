import os
import numpy as np
import lmdb
import caffe

import IPython

num_images = 5000

# Set up model
base_path = '/Users/dpaiton/Work/LANL/PetaVision/Projects/SparseSampling/python/'
model_deploy = os.path.join(base_path, 'mlp_test_deploy.prototxt')
pretrained_weights = os.path.join(base_path, 'checkpoints/mlp_sparse_v1_iter_100000.caffemodel')
net = caffe.Net(model_deploy, pretrained_weights, caffe.TEST)

(nf, ny, nx) = net.blobs['data'].data[0].shape
datasets = ['lmdb_datasets/pvCifar_test_lmdb_v2_samples_'+str(x) for x in range(1,51)]

entropy = np.zeros((len(datasets), num_images))
data = np.zeros((len(datasets), num_images, nf, ny, nx))
labels = np.zeros(num_images)

for datasetIdx, dataset in enumerate(datasets):
    print 'Computing network output entropy for dataset '+dataset
    env = lmdb.open(dataset, readonly=True)
    datum = caffe.proto.caffe_pb2.Datum()

    #Load each frame in dataset
    with env.begin() as txn:
        for key, value in txn.cursor():
            if int(key) == num_images:
                break
            datum.ParseFromString(value)
            flat_x = np.array(datum.float_data)
            data[datasetIdx, int(key), ...] = flat_x.reshape(datum.channels, datum.height, datum.width)
            if datasetIdx == 0:
                labels[int(key)] = datum.label
            else:
                assert(labels[int(key)] == datum.label)

            # Set frame as net input and run net
            net.blobs['data'].data[0] = data[datasetIdx, int(key), ...]
            output = net.forward()

            # Compute entropy of network outputs
            entropy[datasetIdx, int(key)] = -np.sum(np.multiply(output['prob'][0], np.log(output['prob'][0])))

accuracy = np.zeros(len(datasets))
for max_sample in range(len(datasets)):
    print 'Computing accuracy for '+str(max_sample)+' samples.'
    min_entropy_sample_idx = np.argmin(entropy[0:max_sample+1,:], axis=0) # compute min across samples
    for img_idx in range(num_images):
        sample_idx = min_entropy_sample_idx[img_idx]
        net.blobs['data'].data[0] = data[sample_idx, img_idx, ...]
        output = net.forward()
        if int(labels[img_idx]) == np.argmax(output['prob']):
            accuracy[max_sample] += 1.0
    accuracy[max_sample] /= float(num_images)

np.save('accuracyNoAverage.npy', accuracy)
