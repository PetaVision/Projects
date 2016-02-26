#!/usr/bin/env sh

GLOG_logtostderr=0 GLOG_log_dir=checkpoints/log/ /Users/dpaiton/Work/Libraries/caffe/build/tools/caffe test \
    --model=mlp_test.prototxt \
    --weights=checkpoints/mlp_sparse_iter_100000.caffemodel \
    --iterations 100
