# Auctionformer Codes

Auctionformer: A Unified Deep Learning Algorithm for Solving Optimal Strategies in Auction Games


## Run Auctionformer
The dataset is prepared in `codes/dataset.zip`, please unzip the compressed file via:
```shell
cd codes
unzip dataset.zip
```

Then you can train Auctionformer with:
```shell
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--model_name bert --gpu 0
```

To change the pretrain model, please set the argument `"model_name"` with different model's name, such as gpt-2:
```shell
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 1 --use_wandb 1 --log_results 0 \
--model_name gpt-2 --gpu 1
```

And you can train Auctionformer without pretrain model's parameter updating, by setting `"pretrain_requires_grad"` to zero:
```shell
python3 main.py --mechanisms first second --max_entry 3 --distributions uniform gaussian --start_from_zero 0 \
--iterate_game_types 1 --combine_zero 1 --pretrain_requires_grad 0 --use_wandb 1 --log_results 0 \
--model_name bert --gpu 2
```

## Evaluation
By default, the trained Auctionformer will be saved in `./codes/model_ckpt/`, with different model name representing various training arguments.

To evaluate the trained model's performance on different player number, mechanism and distribution, by replacing `{saved_model_name}` with your model:

```shell
python3 evaluate.py --ckpt_name {saved_model_name} --entry_emb 4 --test_mode split
```

Or to check the running time:
```shell
# cpu
python3 evaluate.py --ckpt_name {saved_model_name}  --entry_emb 4 --test_mode symmetric_time --force_cpu 1
# gpu
python3 evaluate.py --ckpt_name {saved_model_name}  --entry_emb 4 --test_mode symmetric_time --force_cpu 0
```

You can also compare the model's predicted result with anaylitic solution in symmetric auctions:
```shell
python3 evaluate.py --mechanisms first --ckpt_name {saved_model_name} --entry_emb 4 --test_mode symmetric
```

## Few-shot Learning

We also provide few-shot experiment for our model.