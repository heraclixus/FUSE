# nohup python main.py --gpu_id 1 --dataset science --run_gumbel_box --embed_size 100 &> log-12-08-run_gumbel_box-science-1.txt & 
# nohup python main.py --gpu_id 0 --dataset science --run_gumbel_box --embed_size 100 &> log-12-08-run_gumbel_box-science-2.txt & 
# nohup python main.py --gpu_id 1 --dataset science --run_gumbel_box --embed_size 100 &> log-12-08-run_gumbel_box-science-3.txt & 
# nohup python main.py --gpu_id 2 --dataset science --run_gumbel_box --embed_size 100 &> log-12-08-run_gumbel_box-science-4.txt & 
# nohup python main.py --gpu_id 3 --dataset science --run_gumbel_box --embed_size 100 &> log-12-08-run_gumbel_box-science-5.txt & 

# nohup python main.py --gpu_id 4 --dataset environment --run_gumbel_box --embed_size 100 &> log-12-08-run_gumbel_box-env-1.txt & 
# nohup python main.py --gpu_id 5 --dataset environment --run_gumbel_box --embed_size 100 &> log-12-08-run_gumbel_box-env-2.txt & 
# nohup python main.py --gpu_id 6 --dataset environment --run_gumbel_box --embed_size 100 &> log-12-08-run_gumbel_box-env-3.txt & 
# nohup python main.py --gpu_id 7 --dataset environment --run_gumbel_box --embed_size 100 &> log-12-08-run_gumbel_box-env-4.txt & 
# nohup python main.py --gpu_id 0 --dataset environment --run_gumbel_box --embed_size 100 &> log-12-08-run_gumbel_box-env-5.txt & 


# nohup python main.py --gpu_id 1 --dataset science --run_gumbel_box --embed_size 500 &> log-12-09-run_gumbel_box-science-1.txt & 
# nohup python main.py --gpu_id 2 --dataset science --run_gumbel_box --embed_size 500 &> log-12-09-run_gumbel_box-science-2.txt & 
# nohup python main.py --gpu_id 3 --dataset science --run_gumbel_box --embed_size 500 &> log-12-09-run_gumbel_box-science-3.txt & 
# nohup python main.py --gpu_id 4 --dataset science --run_gumbel_box --embed_size 500 &> log-12-09-run_gumbel_box-science-4.txt & 
# nohup python main.py --gpu_id 5 --dataset science --run_gumbel_box --embed_size 500 &> log-12-09-run_gumbel_box-science-5.txt & 

# nohup python main.py --gpu_id 6 --dataset environment --run_gumbel_box --embed_size 500 &> log-12-09-run_gumbel_box-env-1.txt & 
# nohup python main.py --gpu_id 7 --dataset environment --run_gumbel_box --embed_size 500 &> log-12-09-run_gumbel_box-env-2.txt & 
# nohup python main.py --gpu_id 0 --dataset environment --run_gumbel_box --embed_size 500 &> log-12-09-run_gumbel_box-env-3.txt & 
# nohup python main.py --gpu_id 1 --dataset environment --run_gumbel_box --embed_size 500 &> log-12-09-run_gumbel_box-env-4.txt & 
# nohup python main.py --gpu_id 2 --dataset environment --run_gumbel_box --embed_size 500 &> log-12-09-run_gumbel_box-env-5.txt & 


# nohup python main.py --gpu_id 0 --dataset environment --run_gumbel_box --embed_size 100 --strength_alpha 0.1 &> log-12-07-run_gumbel_box-env-1.txt & 
# nohup python main.py --gpu_id 7 --dataset environment --run_gumbel_box --embed_size 100 --strength_alpha 1.0 &> log-12-07-run_gumbel_box-env-2.txt & 
# nohup python main.py --gpu_id 3 --dataset environment --run_gumbel_box --embed_size 100 --strength_alpha 0.0 &> log-12-07-run_gumbel_box-env-3.txt & 
# nohup python main.py --gpu_id 3 --dataset environment --run_gumbel_box --strength_alpha 1.0 &> log-12-06-run_gumbel_box-env-4.txt &


# study gumbel box 
# nohup python main.py --gpu_id 0 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type ratio --box_score_mode whole &> log-env-12-08-gumbel-rw.txt &
# nohup python main.py --gpu_id 1 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type ratio --box_score_mode whole --strength_alpha 1.0 &> log-env-12-08-gumbel-rw-alpha1.0.txt &
# nohup python main.py --gpu_id 2 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type ratio --box_score_mode average &> log-env-12-08-gumbel-ra.txt &
# nohup python main.py --gpu_id 3 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type ratio --box_score_mode average --strength_alpha 1.0 &> log-env-12-08-gumbel-ra-alpha1.0.txt &

# nohup python main.py --gpu_id 4 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type diff --box_score_mode whole &> log-env-12-08-gumbel-dw.txt &
# nohup python main.py --gpu_id 5 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type diff --box_score_mode whole --strength_alpha 1.0 &> log-env-12-08-gumbel-dw-alpha1.0.txt &
# nohup python main.py --gpu_id 6 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type diff --box_score_mode average &> log-env-12-08-gumbel-da.txt &
# nohup python main.py --gpu_id 7 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type diff --box_score_mode average --strength_alpha 1.0 &> log-env-12-08-gumbel-da-alpha1.0.txt &


# nohup python main.py --gpu_id 0 --epochs 200 --lr_projection 1e-4 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type ratio --box_score_mode whole &> log-env-12-09-gumbel-rw.txt &
# nohup python main.py --gpu_id 1 --epochs 200 --lr_projection 1e-4 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type ratio --box_score_mode whole --strength_alpha 1.0 &> log-env-12-09-gumbel-rw-alpha1.0.txt &
# nohup python main.py --gpu_id 2 --epochs 200 --lr_projection 1e-4 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type ratio --box_score_mode average &> log-env-12-09-gumbel-ra.txt &
# nohup python main.py --gpu_id 3 --epochs 200 --lr_projection 1e-4 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type ratio --box_score_mode average --strength_alpha 1.0 &> log-env-12-09-gumbel-ra-alpha1.0.txt &

# nohup python main.py --gpu_id 4 --epochs 200 --lr_projection 1e-4 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type diff --box_score_mode whole &> log-env-12-09-gumbel-dw.txt &
# nohup python main.py --gpu_id 5 --epochs 200 --lr_projection 1e-4 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type diff --box_score_mode whole --strength_alpha 1.0 &> log-env-12-09-gumbel-dw-alpha1.0.txt &
# nohup python main.py --gpu_id 6 --epochs 200 --lr_projection 1e-4 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type diff --box_score_mode average &> log-env-12-09-gumbel-da.txt &
# nohup python main.py --gpu_id 7 --epochs 200 --lr_projection 1e-4 --dataset environment --run_gumbel_box --embed_size 100 --box_score_type diff --box_score_mode average --strength_alpha 1.0 &> log-env-12-09-gumbel-da-alpha1.0.txt &
