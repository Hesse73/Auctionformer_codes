# Auctionformer Codes

Auctionformer: A Unified Deep Learning Algorithm for Solving Optimal Strategies in Auction Games


## Run Auctionformer
The dataset is prepared in `./codes/dataset.zip`, please unzip the compressed file:
```sh
cd ./codes
unzip dataset.zip
```

Then you can train Auctionformer with:
```sh
cd ./codes
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--model_name bert --gpu 0
```

To change the pretrain model, please set the argument `"model_name"` with different model's name, such as gpt-2:
```sh
cd ./codes
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--model_name gpt-2 --gpu 1
```

And you can train Auctionformer without pretrain model's parameter updating, by setting `"pretrain_requires_grad"` to zero:
```sh
cd ./codes
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 0 --use_wandb 1 --log_results 0 \
--model_name bert --gpu 2
```

## Evaluation
By default, the trained Auctionformer will be saved in `./codes/model_ckpt/`, with different model name representing various training arguments.

To evaluate the trained model's performance on different player number, mechanism and distribution, by replacing `{saved_model_name}` with your model:

```sh
cd ./codes
python3 evaluate.py --ckpt_name {saved_model_name} --entry_emb 4 --test_mode split
```

Or to check the running time:
```sh
cd ./codes
# cpu
python3 evaluate.py --ckpt_name {saved_model_name}  --entry_emb 4 --test_mode symmetric_time --force_cpu 1
# gpu
python3 evaluate.py --ckpt_name {saved_model_name}  --entry_emb 4 --test_mode symmetric_time --force_cpu 0
```

You can also compare the model's predicted result with anaylitic solution in symmetric auctions:
```sh
cd ./codes
python3 evaluate.py --mechanisms first --ckpt_name {saved_model_name} --entry_emb 4 --test_mode symmetric
```

## Few-shot Learning

We also provide few-shot experiment for our model.

Similarly, we prepared another fewshot dataset for training, please unzip the compressed file `./codes/fewshot/dataset.zip` firstly:

```sh
cd ./codes/fewshot
unzip dataset.zip
```

By default, the code will load previously trained model in `./codes/models/`, so please check the model required by the few-shot learning process is already trained and saved.

To fine-tune the model on games with 15/20 players, run with:
```sh
cd ./codes/fewshot
# different player number
# 15 player
python3 main.py --target_mode number --entry_emb 4  --eval_freq 50 --use_dream_booth 0 --batch_size 512 --target_number 15 --gpu 4
python3 main.py --target_mode number --entry_emb 4  --eval_freq 50 --use_dream_booth 1 --batch_size 256 --target_number 15 --gpu 5
# 20 player
python3 main.py --target_mode number --entry_emb 4  --eval_freq 50 --use_dream_booth 0 --batch_size 512 --target_number 20 --gpu 6
python3 main.py --target_mode number --entry_emb 4  --eval_freq 50 --use_dream_booth 1 --batch_size 256 --target_number 20 --gpu 7
```

If you want to test the model's few-shot learnign ability across mechanisms, a trained model on specific mechanism is required. For example, in our paper, we illustrated the result of few-shot from first price to second price, which requires a model trained on only first price mechanism:

```sh
cd ./codes
# run model on FP + U0 (for few-shot test)
python3 main.py --mechanisms first --max_entry 0 --distributions uniform --start_from_zero 1 \
--iterate_game_types 1 --combine_zero 0 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--gpu 4 --lr 1e-5
# run model on FP + G0 (for few-shot test)
python3 main.py --mechanisms first --max_entry 0 --distributions gaussian --start_from_zero 1 \
--iterate_game_types 1 --combine_zero 0 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--gpu 5 --lr 1e-5
# run model on FP + U0G0 (for few-shot test)
python3 main.py --mechanisms first --max_entry 0 --distributions uniform gaussian --start_from_zero 1 \
--iterate_game_types 1 --combine_zero 0 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--gpu 6 --lr 1e-5
```

Then you can fine-tune all of these models on second price mechanisms with:
```sh
cd ./codes/fewshot
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
```