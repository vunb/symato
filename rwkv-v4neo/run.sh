# cd data; wget https://data.deepai.org/enwik8.zip; unzip enwik8.zip; cd ..

python3 train.py --load_model "" --wandb "" --proj_dir "out" \
--data_file "../../data/vlc.xyz" --data_type "symato" \
--ctx_len 512 --epoch_steps 2000 --epoch_count 20 --epoch_begin 0 --epoch_save 5 \
--micro_bsz 16 --n_layer 6 --n_embd 512 --pre_ffn 0 --head_qk 0 \
--lr_init 8e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy ddp_find_unused_parameters_false --grad_cp 0

# python3 train.py --load_model "out/enwik8/rwkv-20.pth" --wandb "" --proj_dir "out/enwik8" \
# --data_file "data/enwik8" --data_type "utf-8" --vocab_size 0 \
# --ctx_len 256 --epoch_steps 3000 --epoch_count 20 --epoch_begin 0 --epoch_save 5 \
# --micro_bsz 24 --n_layer 6 --n_embd 512 --pre_ffn 0 --head_qk 0 \
# --lr_init 8e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
# --accelerator gpu --devices 1 --precision bf16 --strategy ddp_find_unused_parameters_false --grad_cp 0

# Each "epoch" = 2000 steps, 32000 samples, 8192000 tokens
# Data has 99621832 tokens, 6064 vocab size.
# loss=0.881, lr=1.84e-5