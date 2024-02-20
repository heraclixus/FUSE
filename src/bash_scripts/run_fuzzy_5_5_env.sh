margin=$1
n_partition=350
python main.py --gpu_id 3 --seed 0 --dataset environment --margin ${margin} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1013-exp_${margin}_1.txt &
python main.py --gpu_id 2 --seed 1 --dataset environment --margin ${margin} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1013-exp_${margin}_2.txt & 
python main.py --gpu_id 4 --seed 2 --dataset environment --margin ${margin} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1013-exp_${margin}_3.txt &
python main.py --gpu_id 6 --seed 3 --dataset environment --margin ${margin} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1013-exp_${margin}_4.txt &
python main.py --gpu_id 7 --seed 4 --dataset environment --margin ${margin} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset > log-env-1013-exp_${margin}_5.txt
