import numpy as np
import os, types, gc
import torch
from torch.nn import functional as F

from transformers import PreTrainedTokenizerFast
TOKENIZER = PreTrainedTokenizerFast(tokenizer_file="20B_tokenizer.json")

##########################################################################################################
# Step 1: set model & config (use v4 to run your trained-from-scratch models. v4 and v4neo are compatible)
##########################################################################################################

args = types.SimpleNamespace()
args.RUN_DEVICE = "cpu" # 'cuda' // 'cpu' (already fast)
os.environ["RWKV_JIT_ON"] = '1' # '1' or '0'. very useful for GPU/CPU fp32
os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE

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

########################################################################################################

print(f'\nUsing {args.RUN_DEVICE.upper()}. Loading {args.MODEL_NAME}...')

from model_run_f32 import RWKV_RNN
model = RWKV_RNN(args)

out, _ = model.forward([187], None) # warm-up; token_id 187 => '\n'
gc.collect(); torch.cuda.empty_cache() # free mem


def sample_logits(out, temperature=1.0, top_p_usual=None):
    probs = F.softmax(out, dim=-1)
    top_p = top_p_usual
    probs = probs.cpu().numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0: probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    return np.random.choice(a=len(probs), p=probs)

########################################################################################################

src_ctx = TOKENIZER.encode(context) # context => token_ids
src_len = len(src_ctx)
print("\nYour prompt has " + str(src_len) + " tokens.")

init_state = None
init_out = None

# Khởi tạo init_state, init_out cho lần chạy đầu tiên (TRIAL == 0)
for i in range(src_len):
    x = src_ctx[: i + 1]
    if i == src_len - 1: init_out, init_state = model.forward(x, init_state)
    else: init_state = model.forward(x, init_state, preprocess_only=True)
gc.collect(); torch.cuda.empty_cache() # free mem

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ trial {TRIAL} ]------------------------------------------------\n' + context, end="")

    ctx = src_ctx.copy()
    out_last = src_len
    out = init_out.clone()
    state = init_state.clone()

    for i in range(src_len, src_len + LENGTH_PER_TRIAL): # sinh thêm LENGTH_PER_TRIAL
        input_token_ids = ctx[: i + 1][-args.ctx_len:] # lấy args.ctx_len last tokens
        out, state = model.forward(input_token_ids, state)
        out[0] = -999999999  # disable <|endoftext|>

        next_token_id = sample_logits(out, TEMPERATURE, top_p) # lấy mẫu ngẫu nhiên token tiếp theo
        ctx.append(next_token_id)

        token = TOKENIZER.decode(ctx[out_last:])
        if '\ufffd' not in token: # is valid utf8 string? (bỏ qua invalid token)
            print(token, end="", flush=True)
            out_last = i+1
