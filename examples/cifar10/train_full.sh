#!/usr/bin/env sh
if [ ! -n "$1" ] ;then
    echo "$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi


TOOLS=./build/tools

GLOG_log_dir=examples/cifar10/log $TOOLS/caffe train --gpu $gpu \
    --solver=examples/cifar10/cifar10_full.solver
