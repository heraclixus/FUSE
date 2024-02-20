n_partition=$1
# python main.py --gpu_id 0 --seed 0 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-1207-exp_${n_partition}_1.txt &
# python main.py --gpu_id 1 --seed 1 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-1207-exp_${n_partition}_2.txt & 
# python main.py --gpu_id 2 --seed 2 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-1207-exp_${n_partition}_3.txt &
# python main.py --gpu_id 0 --seed 3 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-1207-exp_${n_partition}_4.txt &
# python main.py --gpu_id 1 --seed 4 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-1207-exp_${n_partition}_5.txt & 


# n_partition=$1
# python main.py --gpu_id 2 --seed 0 --dataset environment --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1207-exp_${n_partition}_1.txt &
python main.py --gpu_id 1 --seed 1 --dataset environment --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1207-exp_${n_partition}_2.txt & 
python main.py --gpu_id 2 --seed 2 --dataset environment --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1207-exp_${n_partition}_3.txt &
python main.py --gpu_id 3 --seed 3 --dataset environment --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1207-exp_${n_partition}_4.txt &
python main.py --gpu_id 4 --seed 4 --dataset environment --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1207-exp_${n_partition}_5.txt & 
# # sleep 10m