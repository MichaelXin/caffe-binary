#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_log_dir=examples/cifar10_binary/log $TOOLS/caffe train --gpu=$1 \
        --solver=examples/cifar10_binary/nobias.solver \
        --weights=examples/cifar10/cifar10_full_nobias_iter_40000.caffemodel
