#!/bin/bash

touch log_optim_table.csv
echo "dataset,acc,mrr,wp" > log_optim_table.csv

for file in ./log-ulti*.txt
do
    echo $(basename $file)
    filename=$(basename -- "$file")
    acc=$(cat $filename | grep "score:" | cut -d " " -f 3)
    mrr=$(cat $filename | grep "score:" | cut -d " " -f 4 | cut -d ":" -f 2)
    wp=$(cat $filename | grep "score:" | cut -d " " -f 5 | cut -d ":" -f 2)
    echo "science,${acc},${mrr},${wp}" >> log_optim_table.csv
done


for file in ./log-env-ulti*.txt
do
    echo $(basename $file)
    filename=$(basename -- "$file")
    acc=$(cat $filename | grep "score:" | cut -d " " -f 3)
    mrr=$(cat $filename | grep "score:" | cut -d " " -f 4 | cut -d ":" -f 2)
    wp=$(cat $filename | grep "score:" | cut -d " " -f 5 | cut -d ":" -f 2)
    echo "environment,${acc},${mrr},${wp}" >> log_optim_table.csv
done
