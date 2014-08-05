#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    multipie_solver.prototxt 0 2layers_split2_big_all_ip_norm/multipie_train_iter_9000.solverstate

echo "Done."
