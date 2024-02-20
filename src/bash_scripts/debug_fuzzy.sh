nohup python main.py --gpu_id 2 --dataset science --test_only &> log-debug1.txt &

nohup python main.py --gpu_id 3 --dataset science --test_only --use_fuzzyset &> log-debug2.txt &