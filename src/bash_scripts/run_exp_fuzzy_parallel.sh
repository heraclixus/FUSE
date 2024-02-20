#!/bin/bash

for n_partition in 100 150 200 250 300 350 400 450 500 550
do
    nohup bash run_fuzzy1.sh $n_partition
done