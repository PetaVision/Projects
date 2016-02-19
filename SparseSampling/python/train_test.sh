#!/usr/bin/env sh

GLOG_logtostderr=0 GLOG_log_dir=checkpoints/log/ /Users/dpaiton/Work/Libraries/caffe/build/tools/caffe train \
    --solver=solver.prototxt 
    #--weights=checkpoints/mlp_iter_30000.caffemodel
