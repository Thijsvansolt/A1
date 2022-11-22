#!/bin/bash

for i in 1000, 10000, 100000, 1000000, 10000000
do
    echo "Running for $i"
    prun -v -np 1 -native "-C TitanX --gres=gpu:1" ./assign2_1 $i 100000 32 >> scale_results.txt
done

for i in 32, 64, 128, 256, 512, 1024
do
    for j in 1000, 10000, 100000, 1000000, 10000000
    do
        echo "Running for $i $j" >> block_size_results.txt
        prun -v -np 1 -native "-C TitanX --gres=gpu:1" ./assign2_1 $j 100000 $i >> block_size_results.txt
    done
done