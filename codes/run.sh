# run bert
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--model_name bert --gpu 0

# run bert small (change --model_name for different models)
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--model_name bert_small --gpu 1

# run bert without attention
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--requires_value 0 --final_mlp 1 --gpu 0

# run model without MLP decoder
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--final_mlp 0 --gpu 2

# run model without value_dist in query
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--query_style test_raw --gpu 0

# run model with random init (off)
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 0 --use_wandb 1 --log_results 0 \
--gpu 3 --from_pretrain 0

# run model on FP + U0 (for few-shot test)
python3 main.py --mechanisms first --max_entry 0 --distributions uniform --start_from_zero 1 \
--iterate_game_types 1 --combine_zero 0 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--gpu 2 --lr 1e-5

# run model on FP + G0 (for few-shot test)
python3 main.py --mechanisms first --max_entry 0 --distributions gaussian --start_from_zero 1 \
--iterate_game_types 1 --combine_zero 0 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--gpu 5 --lr 1e-5

# run model on FP + U0G0 (for few-shot test)
python3 main.py --mechanisms first --max_entry 0 --distributions uniform gaussian --start_from_zero 1 \
--iterate_game_types 1 --combine_zero 0 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--gpu 6 --lr 1e-5
