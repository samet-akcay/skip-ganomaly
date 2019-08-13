#!/bin/bash

# Run MNIST experiment for each individual dataset.
# For each anomalous digit
for i in {0..9}
do
    echo "Running mnist_$i"
    python train.py --dataset mnist --isize 32 --nc 1 --niter 15 --abnormal_class $i --model ganomaly --display
done
exit 0
