#!/bin/bash

for i in $(seq 6 10); do
    for j in $(seq 10 10 30); do
        THEANO_FLAGS=device=gpu,floatX=float32 python convnet.py 10 3 "full" $i $j;
    done
done
