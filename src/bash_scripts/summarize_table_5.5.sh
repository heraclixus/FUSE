#!/bin/bash

# touch log_margin_table_science.csv
# echo "margin,acc,mrr,wp" > log_margin_table_science.csv

# for file in ./log-1013-*.txt
# do
#     echo $(basename $file)
#     filename=$(basename -- "$file")
#     margin=$(echo $filename | cut -d "_" -f 2)
#     acc=$(cat $filename | grep "score:" | cut -d " " -f 3)
#     mrr=$(cat $filename | grep "score:" | cut -d " " -f 4 | cut -d ":" -f 2)
#     wp=$(cat $filename | grep "score:" | cut -d " " -f 5 | cut -d ":" -f 2)
#     echo "${margin},${acc},${mrr},${wp}" >> log_margin_table_science.csv
# done


touch log_margin_table_environment.csv
echo "margin,acc,mrr,wp" > log_margin_table_environment.csv

for file in ./log-env-1013-*.txt
do
    echo $(basename $file)
    filename=$(basename -- "$file")
    reg=$(echo $filename | cut -d "_" -f 2)
    acc=$(cat $filename | grep "score:" | cut -d " " -f 3)
    mrr=$(cat $filename | grep "score:" | cut -d " " -f 4 | cut -d ":" -f 2)
    wp=$(cat $filename | grep "score:" | cut -d " " -f 5 | cut -d ":" -f 2)
    echo "${reg},${acc},${mrr},${wp}" >> log_margin_table_environment.csv
done
