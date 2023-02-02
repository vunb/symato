import numpy as np
import os, sys, types, time, gc
import torch
from tknz import TOKENIZER

try: os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except: pass

##########################################################################################################
# Step 1: set model & config (use v4 to run your trained-from-scratch models. v4 and v4neo are compatible)
##########################################################################################################

args = types.SimpleNamespace()
args.RUN_DEVICE = "cpu" # 'cuda' // 'cpu' (already fast)
os.environ["RWKV_JIT_ON"] = '1' # '1' or '0'. very useful for GPU/CPU fp32
os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE

TOKEN_MODE = "pile"
WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None

# wget https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth
args.MODEL_NAME = "RWKV-4-Pile-169M-20220807-8023"
args.n_layer = 12
args.n_embd = 768
args.ctx_len = 1024
args.vocab_size = 50277
args.grad_cp = 0
args.my_pos_emb = 0 # không dùng positional embedding

########################################################################################################
# Step 2: set prompt & sampling stuffs
########################################################################################################

context = """\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously 
unexplored valley, in Tibet. Even more surprising to the researchers was the fact 
that the dragons spoke perfect Chinese."""

NUM_TRIALS = 3
LENGTH_PER_TRIAL = 222

TEMPERATURE = 1.0
top_p = 0.8
top_p_newline = 0.9  # only used in TOKEN_MODE = char

########################################################################################################

print(f'\nUsing {args.RUN_DEVICE.upper()}. Loading {args.MODEL_NAME}...')

from model_run_f32 import RWKV_RNN
model = RWKV_RNN(args)

out, _ = model.forward([187], None) # warm-up; token_id 187 => '\n'
gc.collect(); torch.cuda.empty_cache()

print(f'\nLoading tokenizer {WORD_NAME}...')
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
assert TOKEN_MODE == "pile" and tokenizer.tokenizer.decode([187]) == '\n'

########################################################################################################

ctx = tokenizer.tokenizer.encode(context)
src_len = len(ctx)
src_ctx = ctx.copy()

print("\nYour prompt has " + str(src_len) + " tokens.")

init_state = None
init_out = None
state = None
out = None

# TRIAL == 0:
for i in range(src_len):
    x = ctx[: i + 1]
    if i == src_len - 1: init_out, init_state = model.forward(x, init_state)
    else: init_state = model.forward(x, init_state, preprocess_only=True)
gc.collect()
torch.cuda.empty_cache()

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ trial {TRIAL} ]------------------------------------------------\n' + context, end="")

    time_ref = time.time_ns()
    ctx = src_ctx.copy()
    out_last = src_len
    for i in range(src_len, src_len + LENGTH_PER_TRIAL):
        x = ctx[: i + 1]
        x = x[-args.ctx_len:]

        if i == src_len:
            out = init_out.clone()
            state = init_state.clone()
        else:
            out, state = model.forward(x, state)
        if TOKEN_MODE == "pile":
            out[0] = -999999999  # disable <|endoftext|>

        ttt = tokenizer.sample_logits(
            out, x, args.ctx_len,
            temperature=TEMPERATURE,
            top_p_usual=top_p,
            top_p_newline=top_p_newline,
        )
        ctx += [ttt]

        if tokenizer.charMode:
            char = tokenizer.itos[ttt]
            print(char, end="", flush=True)
        else:
            char = tokenizer.tokenizer.decode(ctx[out_last:])
            if '\ufffd' not in char: # is valid utf8 string?
                print(char, end="", flush=True)
                out_last = i+1
