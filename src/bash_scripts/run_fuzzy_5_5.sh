margin=$1
n_partition=500
python main.py --gpu_id 2 --seed 0 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --margin ${margin} --strength_alpha 0.5 --use_fuzzyset &> log-1013-exp_${margin}_1.txt &
python main.py --gpu_id 3 --seed 1 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --margin ${margin} --use_fuzzyset &> log-1013-exp_${margin}_2.txt & 
python main.py --gpu_id 4 --seed 2 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --margin ${margin} --use_fuzzyset &> log-1013-exp_${margin}_3.txt &
python main.py --gpu_id 6 --seed 3 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --margin ${margin} --use_fuzzyset &> log-1013-exp_${margin}_4.txt &
python main.py --gpu_id 7 --seed 4 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --margin ${margin} --use_fuzzyset > log-1013-exp_${margin}_5.txt