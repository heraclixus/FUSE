regularizer=$1 

n_partition=500
# python main.py --gpu_id 3 --seed 0 --dataset science --use_volume_weights --regularize_volume --partition_reg_type ${regularizer} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-1011-exp_${regularizer}_1.txt &
# python main.py --gpu_id 2 --seed 1 --dataset science --use_volume_weights --regularize_volume --partition_reg_type ${regularizer} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-1011-exp_${regularizer}_2.txt & 
# python main.py --gpu_id 4 --seed 2 --dataset science --use_volume_weights --regularize_volume --partition_reg_type ${regularizer} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-1011-exp_${regularizer}_3.txt &
# python main.py --gpu_id 6 --seed 3 --dataset science --use_volume_weights --regularize_volume --partition_reg_type ${regularizer} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-1011-exp_${regularizer}_4.txt &
# python main.py --gpu_id 7 --seed 4 --dataset science --use_volume_weights --regularize_volume --partition_reg_type ${regularizer} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset > log-1011-exp_${regularizer}_5.txt


python main.py --gpu_id 3 --seed 0 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type ${regularizer} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1011-exp_${regularizer}_1.txt &
python main.py --gpu_id 2 --seed 1 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type ${regularizer} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1011-exp_${regularizer}_2.txt & 
python main.py --gpu_id 4 --seed 2 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type ${regularizer} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1011-exp_${regularizer}_3.txt &
python main.py --gpu_id 6 --seed 3 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type ${regularizer} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset &> log-env-1011-exp_${regularizer}_4.txt &
python main.py --gpu_id 7 --seed 4 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type ${regularizer} --score_type weighted_cos --n_partitions ${n_partition} --use_fuzzyset > log-env-1011-exp_${regularizer}_5.txt
# sleep 10m

# python main.py --gpu_id 3 --seed 0 --dataset science --use_volume_weights --regularize_volume --partition_reg_type sigmoid --score_type weighted_cos --n_partitions 500 --use_fuzzyset &> log-1011-exp_sigmoid_1.txt &
