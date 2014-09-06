#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 GLOG_log_dir=4layers_v14_v3 $TOOLS/train_net.bin ./4layers_v14_v3/cfw_solver.prototxt 1

echo "Done."
