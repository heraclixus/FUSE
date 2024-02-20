
n_partition=300

python main.py --gpu_id 1 --seed 0 --dataset environment --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1010-exp_${n_partition}_1.txt &
python main.py --gpu_id 2 --seed 1 --dataset environment --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1010-exp_${n_partition}_2.txt & 
python main.py --gpu_id 4 --seed 2 --dataset environment --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1010-exp_${n_partition}_3.txt &
python main.py --gpu_id 6 --seed 3 --dataset environment --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1010-exp_${n_partition}_4.txt &
python main.py --gpu_id 7 --seed 4 --dataset environment --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset > log-env-1010-exp_${n_partition}_5.txt
sleep 3m