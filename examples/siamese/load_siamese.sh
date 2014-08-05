#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/finetune_net.bin siamese_solver.prototxt ../cfw/4layers/cfw_iter_19100

echo "Done."
