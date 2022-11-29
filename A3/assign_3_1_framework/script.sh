#!/bin/bash

# echo "Running 1 node with 8 processes"
# for i in 1000, 10000, 100000, 1000000, 10000000
# do
#     prun v -np 1 -8 -sge-script $PRUN_ETC/prun-openmpi ./assign3_1 $i 10000 >> 1_8_results.txt
# done

# echo "Running 8 node with 1 process"
# for i in 1000, 10000, 100000, 1000000, 10000000
# do
#     prun v -np 8 -1 -sge-script $PRUN_ETC/prun-openmpi ./assign3_1 $i 10000 >> 8_1_results.txt
# done

# echo "Running 8 node with 8 processes"
# for i in 1000, 10000, 100000, 1000000, 10000000
# do
#     prun v -np 8 -8 -sge-script $PRUN_ETC/prun-openmpi ./assign3_1 $i 10000 >> 8_8_results.txt
# done

echo "Running 1 node with 1 processes"
for i in 1000, 10000, 100000, 1000000, 10000000
do
    prun v -np 1 -1 -sge-script $PRUN_ETC/prun-openmpi ./assign3_1 $i 10000 >> 1_1_results.txt
done