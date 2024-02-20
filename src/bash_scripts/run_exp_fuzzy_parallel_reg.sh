#!/bin/bash

for margin in 0.1 0.15 0.2 0.25
do 
    nohup bash run_fuzzy_5_5.sh $margin
done

for margin in 0.1 0.15 0.2 0.25
do 
    nohup bash run_fuzzy_5_5_env.sh $margin
done