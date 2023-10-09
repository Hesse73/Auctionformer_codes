# different player number
# 15 player
python3 main.py --target_mode number --entry_emb 4  --eval_freq 50 --use_dream_booth 0 --batch_size 512 --target_number 15 --gpu 4
python3 main.py --target_mode number --entry_emb 4  --eval_freq 50 --use_dream_booth 1 --batch_size 256 --target_number 15 --gpu 5
# 20 player
python3 main.py --target_mode number --entry_emb 4  --eval_freq 50 --use_dream_booth 0 --batch_size 512 --target_number 20 --gpu 6
python3 main.py --target_mode number --entry_emb 4  --eval_freq 50 --use_dream_booth 1 --batch_size 256 --target_number 20 --gpu 7
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
