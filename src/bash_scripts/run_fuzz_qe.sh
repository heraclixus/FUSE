nohup python main.py --gpu_id 0 --dataset science --use_fuzzyset --use_fuzzqe --n_partitions 1000 &> log-12-07-run_fuzzqe-science-1.txt & 
nohup python main.py --gpu_id 1 --dataset science --use_fuzzyset --use_fuzzqe --n_partitions 1000 &> log-12-07-run_fuzzqe-science-2.txt & 
nohup python main.py --gpu_id 2 --dataset science --use_fuzzyset --use_fuzzqe --n_partitions 1000 &> log-12-07-run_fuzzqe-science-3.txt & 
nohup python main.py --gpu_id 3 --dataset science --use_fuzzyset --use_fuzzqe --n_partitions 1000 &> log-12-07-run_fuzzqe-science-4.txt & 
nohup python main.py --gpu_id 4 --dataset science --use_fuzzyset --use_fuzzqe --n_partitions 1000 &> log-12-07-run_fuzzqe-science-5.txt & 

nohup python main.py --gpu_id 0 --dataset environment --use_fuzzyset --use_fuzzqe --n_partitions 1000 &> log-12-07-run_fuzzqe-env-1.txt & 
nohup python main.py --gpu_id 1 --dataset environment --use_fuzzyset --use_fuzzqe --n_partitions 1000 &> log-12-07-run_fuzzqe-env-2.txt & 
nohup python main.py --gpu_id 2 --dataset environment --use_fuzzyset --use_fuzzqe --n_partitions 1000  &> log-12-07-run_fuzzqe-env-3.txt & 
nohup python main.py --gpu_id 3 --dataset environment --use_fuzzyset --use_fuzzqe --n_partitions 1000 &> log-12-07-run_fuzzqe-env-4.txt & 
nohup python main.py --gpu_id 4 --dataset environment --use_fuzzyset --use_fuzzqe --n_partitions 1000 &> log-12-07-run_fuzzqe-env-5.txt & 

nohup python main.py --gpu_id 5 --dataset science --use_fuzzyset --use_fuzzqe --use_volume_weights --n_partitions 1000 &> log-12-07-run_fuzzqe-w-science-1.txt & 
nohup python main.py --gpu_id 6 --dataset science --use_fuzzyset --use_fuzzqe --use_volume_weights --n_partitions 1000 &> log-12-07-run_fuzzqe-w-science-2.txt & 
nohup python main.py --gpu_id 7 --dataset science --use_fuzzyset --use_fuzzqe --use_volume_weights --n_partitions 1000 &> log-12-07-run_fuzzqe-w-science-3.txt & 
nohup python main.py --gpu_id 5 --dataset science --use_fuzzyset --use_fuzzqe --use_volume_weights --n_partitions 1000 &> log-12-07-run_fuzzqe-w-science-4.txt & 
nohup python main.py --gpu_id 6 --dataset science --use_fuzzyset --use_fuzzqe --use_volume_weights --n_partitions 1000 &> log-12-07-run_fuzzqe-w-science-5.txt & 

# nohup python main.py --gpu_id 1 --dataset environment --use_fuzzyset --use_volume_weights --use_fuzzqe &> log-12-07-run_fuzzqe-w-env-1.txt & 
# nohup python main.py --gpu_id 2 --dataset environment --use_fuzzyset --use_volume_weights --use_fuzzqe &> log-12-07-run_fuzzqe-w-env-2.txt & 
# nohup python main.py --gpu_id 5 --dataset environment --use_fuzzyset --use_volume_weights --use_fuzzqe &> log-12-07-run_fuzzqe-w-env-3.txt & 
# nohup python main.py --gpu_id 6 --dataset environment --use_fuzzyset --use_volume_weights --use_fuzzqe &> log-12-07-run_fuzzqe-w-env-4.txt & 
# nohup python main.py --gpu_id 7 --dataset environment --use_fuzzyset --use_volume_weights --use_fuzzqe &> log-12-07-run_fuzzqe-w-env-5.txt & 


# nohup python main.py --gpu_id 2 --dataset science --use_fuzzyset --use_fuzzqe --use_volume_weights --regularize_volume &> log-12-06-run_fuzzqe-w1-science-1.txt & 
# nohup python main.py --gpu_id 3 --dataset science --use_fuzzyset --use_fuzzqe --use_volume_weights --regularize_volume &> log-12-06-run_fuzzqe-w1-science-2.txt & 
# nohup python main.py --gpu_id 4 --dataset science --use_fuzzyset --use_fuzzqe --use_volume_weights --regularize_volume &> log-12-06-run_fuzzqe-w1-science-3.txt & 
# nohup python main.py --gpu_id 5 --dataset science --use_fuzzyset --use_fuzzqe --use_volume_weights --regularize_volume &> log-12-06-run_fuzzqe-w1-science-4.txt & 
# nohup python main.py --gpu_id 6 --dataset science --use_fuzzyset --use_fuzzqe --use_volume_weights --regularize_volume &> log-12-06-run_fuzzqe-w1-science-5.txt & 

# nohup python main.py --gpu_id 2 --dataset environment --use_fuzzyset --use_volume_weights --use_fuzzqe --regularize_volume &> log-12-06-run_fuzzqe-w1-env-1.txt & 
# nohup python main.py --gpu_id 3 --dataset environment --use_fuzzyset --use_volume_weights --use_fuzzqe --regularize_volume &> log-12-06-run_fuzzqe-w1-env-2.txt & 
# nohup python main.py --gpu_id 4 --dataset environment --use_fuzzyset --use_volume_weights --use_fuzzqe --regularize_volume &> log-12-06-run_fuzzqe-w1-env-3.txt & 
# nohup python main.py --gpu_id 5 --dataset environment --use_fuzzyset --use_volume_weights --use_fuzzqe --regularize_volume &> log-12-06-run_fuzzqe-w1-env-4.txt & 
# nohup python main.py --gpu_id 6 --dataset environment --use_fuzzyset --use_volume_weights --use_fuzzqe --regularize_volume &> log-12-06-run_fuzzqe-w1-env-5.txt & 


