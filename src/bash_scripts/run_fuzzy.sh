# nohup python main.py --gpu_id 1 --dataset science --use_fuzzyset &> log-1008-exp1.txt & 

# without or without volume weight  
# nohup python main.py --gpu_id 1 --dataset science --score_type weighted_cos --use_fuzzyset &> log-1008-exp2.txt & 
# nohup python main.py --gpu_id 2 --dataset science --score_type weighted_cos --use_volume_weights --use_fuzzyset &> log-1008-exp3.txt & 
# nohup python main.py --gpu_id 3 --dataset science --score_type possibility --use_fuzzyset &> log-1008-exp4.txt & 
# nohup python main.py --gpu_id 4 --dataset science --score_type possibility --use_volume_weights --use_fuzzyset &> log-1008-exp5.txt & 

# volume weight with regularization 
# nohup python main.py --gpu_id 6 --dataset science --score_type weighted_cos --use_volume_weights --regularize_volume --partition_reg_type 01 --use_fuzzyset &> log-1008-exp6.txt & 
# nohup python main.py --gpu_id 7 --dataset science --score_type weighted_cos --use_volume_weights --regularize_volume --partition_reg_type sigmoid --use_fuzzyset &> log-1008-exp7.txt & 
# nohup python main.py --gpu_id 1 --dataset science --score_type possibility --use_volume_weights --regularize_volume --partition_reg_type 01 --use_fuzzyset &> log-1008-exp8.txt & 
# nohup python main.py --gpu_id 2 --dataset science --score_type possibility --use_volume_weights --regularize_volume --partition_reg_type sigmoid --use_fuzzyset &> log-1008-exp9.txt & 


# different regularization for fuzzy set 
# nohup python main.py --gpu_id 3 --dataset science --regularizer_type sigmoid --score_type weighted_cos --use_fuzzyset &> log-1008-exp10.txt & 
# nohup python main.py --gpu_id 4 --dataset science --regularizer_type sigmoid --score_type weighted_cos --use_volume_weights --use_fuzzyset &> log-1008-exp11.txt & 
# nohup python main.py --gpu_id 6 --dataset science --regularizer_type sigmoid --score_type possibility --use_fuzzyset &> log-1008-exp12.txt & 
# nohup python main.py --gpu_id 7 --dataset science --regularizer_type sigmoid --score_type possibility --use_volume_weights --use_fuzzyset &> log-1008-exp13.txt & 

# volume weight with regularization 
# nohup python main.py --gpu_id 6 --dataset science --regularizer_type sigmoid --score_type weighted_cos --use_volume_weights --regularize_volume --partition_reg_type 01 --use_fuzzyset &> log-1008-exp14.txt & 
# nohup python main.py --gpu_id 7 --dataset science --regularizer_type sigmoid --score_type weighted_cos --use_volume_weights --regularize_volume --partition_reg_type sigmoid --use_fuzzyset &> log-1008-exp15.txt & 
# nohup python main.py --gpu_id 1 --dataset science --regularizer_type sigmoid --score_type possibility --use_volume_weights --regularize_volume --partition_reg_type 01 --use_fuzzyset &> log-1008-exp16.txt & 
# nohup python main.py --gpu_id 2 --dataset science --regularizer_type sigmoid --score_type possibility --use_volume_weights --regularize_volume --partition_reg_type sigmoid --use_fuzzyset &> log-1008-exp17.txt & 


# asymmetry based loss: poss 
# nohup python main.py --gpu_id 3 --dataset science --score_type weighted_cos --strength_alpha 0.1 --use_fuzzyset &> log-1008-exp18.txt & 
# nohup python main.py --gpu_id 4 --dataset science --regularizer_type sigmoid --strength_alpha 0.1 --score_type weighted_cos --use_volume_weights --use_fuzzyset &> log-1008-exp19.txt & 

# nohup python main.py --gpu_id 1 --dataset science --score_type weighted_cos --strength_alpha 0.1 --strength_beta 0.01 --use_fuzzyset &> log-1008-exp20.txt & 
# nohup python main.py --gpu_id 2 --dataset science --regularizer_type sigmoid --strength_alpha 0.1 --strength_beta 0.01 --score_type weighted_cos --use_volume_weights --use_fuzzyset &> log-1008-exp21.txt & 

# nohup python main.py --gpu_id 3 --dataset science --score_type weighted_cos --strength_alpha 0.05 --strength_beta 0.01 --use_fuzzyset &> log-1008-exp22.txt & 
# nohup python main.py --gpu_id 4 --dataset science --regularizer_type sigmoid --strength_alpha 0.05 --strength_beta 0.01 --score_type weighted_cos --use_volume_weights --use_fuzzyset &> log-1008-exp23.txt & 

# n_partitions 

# nohup python main.py --gpu_id 6 --dataset science --score_type weighted_cos --n_partitions 150  --use_fuzzyset &> log-1008-exp24.txt & 
# nohup python main.py --gpu_id 7 --dataset science --regularizer_type sigmoid --n_partitions 150 --score_type weighted_cos --use_volume_weights --use_fuzzyset &> log-1008-exp25.txt & 

# nohup python main.py --gpu_id 1 --dataset science --score_type weighted_cos --n_partitions 200  --use_fuzzyset &> log-1008-exp26.txt & 
# nohup python main.py --gpu_id 2 --dataset science --regularizer_type sigmoid --n_partitions 200 --score_type weighted_cos --use_volume_weights --use_fuzzyset &> log-1008-exp27.txt & 

# nohup python main.py --gpu_id 3 --dataset science --score_type weighted_cos --n_partitions 250  --use_fuzzyset &> log-1008-exp28.txt & 
# nohup python main.py --gpu_id 4 --dataset science --regularizer_type sigmoid --n_partitions 250 --score_type weighted_cos --use_volume_weights --use_fuzzyset &> log-1008-exp29.txt & 

# nohup python main.py --gpu_id 6 --dataset science --score_type weighted_cos --n_partitions 300  --use_fuzzyset &> log-1008-exp30.txt & 
# nohup python main.py --gpu_id 7 --dataset science --regularizer_type sigmoid --n_partitions 300 --score_type weighted_cos --use_volume_weights --use_fuzzyset &> log-1008-exp31.txt & 

# nohup python main.py --gpu_id 1 --dataset science --score_type weighted_cos --n_partitions 350  --use_fuzzyset &> log-1008-exp32.txt & 
# nohup python main.py --gpu_id 2 --dataset science --score_type weighted_cos --n_partitions 400  --use_fuzzyset &> log-1008-exp33.txt & 
# nohup python main.py --gpu_id 3 --dataset science --score_type weighted_cos --n_partitions 450  --use_fuzzyset &> log-1008-exp34.txt & 
# nohup python main.py --gpu_id 4 --dataset science --score_type weighted_cos --n_partitions 500  --use_fuzzyset &> log-1008-exp35.txt & 
# nohup python main.py --gpu_id 6 --dataset science --score_type weighted_cos --n_partitions 550  --use_fuzzyset &> log-1008-exp36.txt & 
# nohup python main.py --gpu_id 7 --dataset science --score_type weighted_cos --n_partitions 600  --use_fuzzyset &> log-1008-exp37.txt & 

# weaker asymmetry
# nohup python main.py --gpu_id 1 --dataset science --score_type weighted_cos --n_partitions 250 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp38.txt & 
# nohup python main.py --gpu_id 2 --dataset science --score_type weighted_cos --n_partitions 300 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp39.txt & 
# nohup python main.py --gpu_id 3 --dataset science --score_type weighted_cos --n_partitions 350 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp40.txt & 

# nohup python main.py --gpu_id 4 --dataset science --score_type weighted_cos --n_partitions 250 --strength_alpha 0.05 --use_fuzzyset &> log-1009-exp41.txt & 
# nohup python main.py --gpu_id 6 --dataset science --score_type weighted_cos --n_partitions 300 --strength_alpha 0.05 --use_fuzzyset &> log-1009-exp42.txt & 
# nohup python main.py --gpu_id 7 --dataset science --score_type weighted_cos --n_partitions 350 --strength_alpha 0.05 --use_fuzzyset &> log-1009-exp43.txt & 



# environment dataset
# nohup python main.py --gpu_id 1 --dataset environment --score_type weighted_cos --n_partitions 250 --use_fuzzyset &> log-1009-exp44.txt & 
# nohup python main.py --gpu_id 2 --dataset environment --score_type weighted_cos --n_partitions 300 --use_fuzzyset &> log-1009-exp45.txt & 
# nohup python main.py --gpu_id 4 --dataset environment --score_type weighted_cos --n_partitions 350 --use_fuzzyset &> log-1009-exp46.txt & 

# nohup python main.py --gpu_id 6 --dataset environment --score_type weighted_cos --n_partitions 350 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp47.txt & 
# nohup python main.py --gpu_id 7 --dataset environment --score_type weighted_cos --n_partitions 400 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp48.txt & 
# nohup python main.py --gpu_id 7 --dataset environment --score_type weighted_cos --n_partitions 450 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp49.txt & 

# nohup python main.py --gpu_id 1 --seed 0 --dataset science --score_type weighted_cos --n_partitions 350 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp50.txt & 
# nohup python main.py --gpu_id 2 --seed 1 --dataset science --score_type weighted_cos --n_partitions 350 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp51.txt & 
# nohup python main.py --gpu_id 4 --seed 2 --dataset science --score_type weighted_cos --n_partitions 350 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp52.txt & 
# nohup python main.py --gpu_id 6 --seed 3 --dataset science --score_type weighted_cos --n_partitions 350 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp53.txt & 
# nohup python main.py --gpu_id 7 --seed 4 --dataset science --score_type weighted_cos --n_partitions 350 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp54.txt & 

# nohup python main.py --gpu_id 1 --seed 5 --dataset science --score_type weighted_cos --n_partitions 350 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp55.txt & 
# nohup python main.py --gpu_id 2 --seed 6 --dataset science --score_type weighted_cos --n_partitions 350 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp56.txt & 
# nohup python main.py --gpu_id 4 --seed 7 --dataset science --score_type weighted_cos --n_partitions 350 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp57.txt & 
# nohup python main.py --gpu_id 6 --seed 8 --dataset science --score_type weighted_cos --n_partitions 350 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp58.txt & 
# nohup python main.py --gpu_id 7 --seed 9 --dataset science --score_type weighted_cos --n_partitions 350 --strength_alpha 0.01 --use_fuzzyset &> log-1009-exp59.txt & 



# nohup python main.py --gpu_id 1 --seed 0 --dataset environment --score_type weighted_cos --n_partitions 300 --use_fuzzyset &> log-1009-exp60.txt & 
# nohup python main.py --gpu_id 2 --seed 1 --dataset environment --score_type weighted_cos --n_partitions 300 --use_fuzzyset &> log-1009-exp61.txt & 
# nohup python main.py --gpu_id 4 --seed 2 --dataset environment --score_type weighted_cos --n_partitions 300 --use_fuzzyset &> log-1009-exp62.txt & 
# nohup python main.py --gpu_id 6 --seed 3 --dataset environment --score_type weighted_cos --n_partitions 300 --use_fuzzyset &> log-1009-exp63.txt & 
# nohup python main.py --gpu_id 7 --seed 4 --dataset environment --score_type weighted_cos --n_partitions 300 --use_fuzzyset &> log-1009-exp64.txt & 

# nohup python main.py --gpu_id 1 --seed 5 --dataset environment --score_type weighted_cos --n_partitions 300 --use_fuzzyset &> log-1009-exp65.txt & 
# nohup python main.py --gpu_id 2 --seed 6 --dataset environment --score_type weighted_cos --n_partitions 300 --use_fuzzyset &> log-1009-exp66.txt & 
# nohup python main.py --gpu_id 4 --seed 7 --dataset environment --score_type weighted_cos --n_partitions 300 --use_fuzzyset &> log-1009-exp67.txt & 
# nohup python main.py --gpu_id 6 --seed 8 --dataset environment --score_type weighted_cos --n_partitions 300 --use_fuzzyset &> log-1009-exp68.txt & 
# nohup python main.py --gpu_id 7 --seed 9 --dataset environment --score_type weighted_cos --n_partitions 300 --use_fuzzyset &> log-1009-exp69.txt &