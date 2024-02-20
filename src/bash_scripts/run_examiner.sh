 # python main.py --gpu_id 4 --seed 3 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 350 --use_fuzzyset &> log-env-ulti-exp_3.txt &
# python main.py --gpu_id 6 --seed 4 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 350 --use_fuzzyset &> log-env-ulti-exp_4.txt &
# python main.py --gpu_id 7 --seed 5 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 350 --use_fuzzyset &> log-env-ulti-exp_5.txt & 


python set_operations.py --gpu_id 1 --seed 1 --dataset science --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --score_type weighted_cos --n_partitions 500 --margin 0.2 --use_fuzzyset &> log-z_set_operations.txt &
