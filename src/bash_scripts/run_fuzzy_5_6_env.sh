python main.py --gpu_id 0 --seed 0 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 1000 --use_fuzzyset &> log-env-1000-exp_1.txt &
python main.py --gpu_id 3 --seed 1 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 1000 --use_fuzzyset &> log-env-1000-exp_2.txt & 
python main.py --gpu_id 4 --seed 2 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 1000 --use_fuzzyset &> log-env-1000-exp_3.txt &
python main.py --gpu_id 5 --seed 3 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 1000 --use_fuzzyset &> log-env-1000-exp_4.txt &
python main.py --gpu_id 6 --seed 4 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 1000 --use_fuzzyset > log-env-1000-exp_5.txt


python main.py --gpu_id 0 --seed 0 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 1500 --use_fuzzyset &> log-env-1500-exp_1.txt &
python main.py --gpu_id 3 --seed 1 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 1500 --use_fuzzyset &> log-env-1500-exp_2.txt & 
python main.py --gpu_id 4 --seed 2 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 1500 --use_fuzzyset &> log-env-1500-exp_3.txt &
python main.py --gpu_id 5 --seed 3 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 1500 --use_fuzzyset &> log-env-1500-exp_4.txt &
python main.py --gpu_id 6 --seed 4 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 1500 --use_fuzzyset > log-env-1500-exp_5.txt

python main.py --gpu_id 0 --seed 0 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 2000 --use_fuzzyset &> log-env-2000-exp_1.txt &
python main.py --gpu_id 3 --seed 1 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 2000 --use_fuzzyset &> log-env-2000-exp_2.txt & 
python main.py --gpu_id 4 --seed 2 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 2000 --use_fuzzyset &> log-env-2000-exp_3.txt &
python main.py --gpu_id 5 --seed 3 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 2000 --use_fuzzyset &> log-env-2000-exp_4.txt &
python main.py --gpu_id 6 --seed 4 --dataset environment --use_volume_weights --regularize_volume --partition_reg_type sigmoid --strength_alpha 0.5 --margin 0.2 --score_type weighted_cos --n_partitions 2000 --use_fuzzyset > log-env-2000-exp_5.txt
