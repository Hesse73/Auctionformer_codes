# run sp+entry
python3 main.py --mechanisms second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 0 --pretrain_requires_grad 1 --use_wandb 0 --dataset_size 2000 \
--lr 1e-5 --gpu 6

# run sp+entry with Embedding_q fixed
python3 main.py --mechanisms second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 0 --pretrain_requires_grad 1 --use_wandb 1 --dataset_size 2000 \
--lr 1e-5 --gpu 6 --fix_emb_q 1

# run player=15 with no entry
python3 main.py --mechanisms first second --max_entry 0 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 0 --pretrain_requires_grad 1 --use_wandb 0 --fix_player 1 --dataset_size 2000 \
--max_player=15 --lr 1e-5 --gpu 5

# run player=15 with embedding_q fixed
python3 main.py --mechanisms first second --max_entry 0 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 0 --pretrain_requires_grad 1 --use_wandb 0 --fix_player 1 --dataset_size 2000 \
--max_player=15 --lr 1e-5 --gpu 5 --fix_emb_q 1

#### paper's result
# fp -> sp
# U0
python3 main.py --target_mode sp --entry_emb 1 --use_dream_booth 0 --batch_size 1024 --target_dist uniform --gpu 0
python3 main.py --target_mode sp --entry_emb 1 --use_dream_booth 1 --batch_size 512 --target_dist uniform --gpu 1
# G0
python3 main.py --target_mode sp --entry_emb 1 --use_dream_booth 0 --batch_size 1024 --target_dist gaussian --gpu 0
python3 main.py --target_mode sp --entry_emb 1 --use_dream_booth 1 --batch_size 512 --target_dist gaussian --gpu 1
# U0G0
python3 main.py --target_mode sp --entry_emb 1 --use_dream_booth 0 --batch_size 1024 --target_dist uniform gaussian --gpu 2
python3 main.py --target_mode sp --entry_emb 1 --use_dream_booth 1 --batch_size 512 --target_dist uniform gaussian --gpu 3
# different number
# 15 player
python3 main.py --target_mode number --entry_emb 4  --eval_freq 50 --use_dream_booth 0 --batch_size 512 --target_number 15 --gpu 4
python3 main.py --target_mode number --entry_emb 4  --eval_freq 50 --use_dream_booth 1 --batch_size 256 --target_number 15 --gpu 5
# 20 player
python3 main.py --target_mode number --entry_emb 4  --eval_freq 50 --use_dream_booth 0 --batch_size 512 --target_number 20 --gpu 6
python3 main.py --target_mode number --entry_emb 4  --eval_freq 50 --use_dream_booth 1 --batch_size 256 --target_number 20 --gpu 7
## fine-tune symmetric first price
#python3 main.py --target_mode sym --max_entry 4 --gpu 7 --mechanisms first --distributions uniform --use_dream_booth 0