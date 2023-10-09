# evaluate main results with bert_on
python3 evaluate.py --ckpt_name 'f|s|e_u|g|nz_iter|+0_n=10_v=20_bert+NV.add_on_Expl_batch=1024_lr=0.0001_update_max_epoch=300' --entry_emb 4 --test_mode split
# evaluate main results with bert_off
python3 evaluate.py --ckpt_name 'f|s|e_u|g|nz_iter|+0_n=10_v=20_bert+NV.add_off_Expl_batch=1024_lr=0.0001_update_max_epoch=300' --entry_emb 4 --test_mode split

# runtime
# cpu
python3 evaluate.py --ckpt_name 'f|s|e_u|g|nz_iter|+0_n=10_v=20_bert+NV.add_on_Expl_batch=1024_lr=0.0001_update_max_epoch=300'  --entry_emb 4 --test_mode symmetric_time --force_cpu 1
# gpu
python3 evaluate.py --ckpt_name 'f|s|e_u|g|nz_iter|+0_n=10_v=20_bert+NV.add_on_Expl_batch=1024_lr=0.0001_update_max_epoch=300'  --entry_emb 4 --test_mode symmetric_time --force_cpu 0

# L2
python3 evaluate.py --mechanisms first --ckpt_name 'f|s|e_u|g|nz_iter|+0_n=10_v=20_bert+NV.add_on_Expl_batch=1024_lr=0.0001_update_max_epoch=300' --entry_emb 4 --test_mode symmetric
