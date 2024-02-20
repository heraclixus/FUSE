lambda=$1

n_partition=500
python main.py --gpu_id 2 --seed 0 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --strength_alpha ${lambda} --use_fuzzyset &> log-1012-exp_${lambda}_0.txt &
python main.py --gpu_id 3 --seed 1 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --strength_alpha ${lambda} --use_fuzzyset &> log-1012-exp_${lambda}_1.txt & 
python main.py --gpu_id 4 --seed 2 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --strength_alpha ${lambda} --use_fuzzyset &> log-1012-exp_${lambda}_2.txt &
python main.py --gpu_id 6 --seed 3 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --strength_alpha ${lambda} --use_fuzzyset &> log-1012-exp_${lambda}_3.txt &
python main.py --gpu_id 7 --seed 4 --dataset science --score_type weighted_cos --n_partitions ${n_partition} --strength_alpha ${lambda} --use_fuzzyset > log-1012-exp_${lambda}_4.txt


# n_partition=350
# python main.py --gpu_id 3 --seed 0 --dataset environment --strength_alpha ${lambda} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1012-exp_${lambda}_1.txt &
# python main.py --gpu_id 2 --seed 1 --dataset environment --strength_alpha ${lambda} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1012-exp_${lambda}_2.txt & 
# python main.py --gpu_id 4 --seed 2 --dataset environment --strength_alpha ${lambda} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1012-exp_${lambda}_3.txt &
# python main.py --gpu_id 6 --seed 3 --dataset environment --strength_alpha ${lambda} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1012-exp_${lambda}_4.txt &
# python main.py --gpu_id 7 --seed 4 --dataset environment --strength_alpha ${lambda} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset > log-env-1012-exp_${lambda}_5.txt