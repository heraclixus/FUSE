#!/bin/bash

touch log_partition_table_environment.csv
echo "n_partitions,acc,mrr,wp" > log_partition_table_environment.csv

for file in ./log-env-1010-*.txt
do
    echo $(basename $file)
    filename=$(basename -- "$file")
    n_partitions=$(echo $filename | cut -d "." -f 1 | cut -d "_" -f 2)
    acc=$(cat $filename | grep "score:" | cut -d " " -f 3)
    mrr=$(cat $filename | grep "score:" | cut -d " " -f 4 | cut -d ":" -f 2)
    wp=$(cat $filename | grep "score:" | cut -d " " -f 5 | cut -d ":" -f 2)
    echo "${n_partitions},${acc},${mrr},${wp}" >> log_partition_table_environment.csv
done