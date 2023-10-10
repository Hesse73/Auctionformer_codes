# run bert
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 1 --use_wandb 0 --log_results 0 \
--model_name bert --gpu 0

# run GPT2 (change --model_name for different models)
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 1 --use_wandb 0 --log_results 0 \
--model_name gpt-2 --gpu 1

# run bert with no pretrain model parameter update
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 0 --use_wandb 0 --log_results 0 \
--model_name bert --gpu 2

# run model on FP + U0 (for few-shot test)
python3 main.py --mechanisms first --max_entry 0 --distributions uniform --start_from_zero 1 \
--iterate_game_types 1 --combine_zero 0 --pretrain_requires_grad 1 --use_wandb 0 --log_results 0 \
--gpu 4 --lr 1e-5

# run model on FP + G0 (for few-shot test)
python3 main.py --mechanisms first --max_entry 0 --distributions gaussian --start_from_zero 1 \
--iterate_game_types 1 --combine_zero 0 --pretrain_requires_grad 1 --use_wandb 0 --log_results 0 \
--gpu 5 --lr 1e-5

# run model on FP + U0G0 (for few-shot test)
python3 main.py --mechanisms first --max_entry 0 --distributions uniform gaussian --start_from_zero 1 \
--iterate_game_types 1 --combine_zero 0 --pretrain_requires_grad 1 --use_wandb 0 --log_results 0 \
--gpu 6 --lr 1e-5
