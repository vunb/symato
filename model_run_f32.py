import numpy as np
import os, types, gc
import torch
from torch.nn import functional as F

########################################################################################################
# Step 0: Define the model in inference mode
########################################################################################################

class RWKV_RNN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.RUN_DEVICE = args.RUN_DEVICE

        # Load params (trọng số) từ file vào vào bộ nhớ (biến w)
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        for k in w.keys():
            if      '.time_' in k: w[k] = w[k].squeeze() # (A,1,B,1) => (A,B)
            if '.time_decay' in k: w[k] = -torch.exp(w[k].float()) # e ^ negative = decay it's actually e ^ ( - e^ x )
            else: w[k] = w[k].float() # convert to fp32 type

            w[k].requires_grad = False # chỉ inference, nên không cần gradient
            if args.RUN_DEVICE == 'cuda' and k != 'emb.weight':
                w[k] = w[k].cuda()           # ^^ embedding lookup table stay in ram

        # Gán trọng số vào tham số mô hình
        self.w = types.SimpleNamespace()
        self.w.blocks = {}
        for k in w.keys():
            parts = k.split('.') # Ví dụ k = "blocks.0.att.value.weight" => parts = ['block','0','att','value','weight']
            last = parts.pop() # => last = weight; parts = ['block','0','att','value']
            here = self.w # độ sâu hiện tại của tham số mô hình
            for i, p in enumerate(parts): # từng bước mở rộng namespace
                if p.isdigit(): # tầng thứ p
                    p = int(p) # dùng [] vì here (w.blocks) là dict object {}
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else: # dùng hasattr, setattr, getattr vì here là types.SimpleNamespace()
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k]) # gán giá trị vào namespace cuối cùng => self.blocks[0].att.value.weight = w[k]

        self.eval() # set torch to inference mode
        gc.collect() # giải phóng ram
        torch.cuda.empty_cache() # giải phóng vram

    '''state[] để lưu trạng thái của rnn, bước chạy thứ i ghi lại 5 trạng thái: 
    i+0 = ffn_xx : token của bước channel-mixing trước 
    i+1 = att_xx : token của bước time-mixing trước
    i+2 = att_aa : exp moving avg của kv 
    i+3 = att_bb : exp moving avg của k
    i+4 = att_pp : use pp to stove exponent of aa and bb
    '''
    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    # @torch.compile
    def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        ffn_xx = 5*i+0 # feed-forward or channel mixing
        # token-shift with diff mixing factors for k and r
        xk = x * time_mix_k + state[ffn_xx] * (1 - time_mix_k)
        xr = x * time_mix_r + state[ffn_xx] * (1 - time_mix_r)
        state[ffn_xx] = x # prev_x = x

        r = torch.sigmoid(rw @ xr) # receptance factor thuộc 0 -> 1
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        return r * (vw @ k)

    # @torch.compile
    def time_mixing(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        att_xx = 5*i+1 # attention or time mixing
        # token-shift
        xk = x * time_mix_k + state[att_xx] * (1 - time_mix_k)
        xv = x * time_mix_v + state[att_xx] * (1 - time_mix_v)
        xr = x * time_mix_r + state[att_xx] * (1 - time_mix_r)
        state[att_xx] = x # prev_x = x

        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv

        aa = state[5*i+2] # exponential moving average of kv
        bb = state[5*i+3] # exponential moving average of k
        pp = state[5*i+4] # idea: use pp to store exponent of a and b

        ww = time_first + k # u + k_i
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b

        ww = pp + time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = qq

        return ow @ (r * wkv)

    def forward(self, token_id, state, preprocess_only = False):
        with torch.no_grad():
            w = self.w
            args = self.args

            x = w.emb.weight[token_id]
            if self.RUN_DEVICE == 'cuda': x = x.cuda()

            if state == None: # khởi tạo trạng thái hệ thống
                state = torch.zeros(args.n_layer * 5, args.n_embd, device=self.RUN_DEVICE)
                for i in range(args.n_layer): state[5*i+4] -= 1e30 # state[att_pp] = dương vô cực

            # Áp dụng layer-norm-0 ở tầng đầu tiên để small-init-emb trick hoạt động
            x = self.layer_norm(x, w.blocks[0].ln0)
            
            for i in range(args.n_layer): # Với mỗi tầng áp dụng:
                # 1/ time-mixing
                att = w.blocks[i].att # trọng số của khối time-mixing
                x = x + self.time_mixing(self.layer_norm(x, w.blocks[i].ln1), state, i, 
                    att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay, 
                    att.key.weight, att.value.weight, att.receptance.weight, att.output.weight)
                
                # 2/ channel-mixing
                ffn = w.blocks[i].ffn # trọng số của khối channel-mixing
                x = x + self.channel_mixing(self.layer_norm(x, w.blocks[i].ln2), state, i, 
                    ffn.time_mix_k, ffn.time_mix_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)

            if preprocess_only: return state
            # Cuối cùng là bộ phân lớp cho ra next token probabilities
            x = self.layer_norm(x, w.ln_out)
            x = w.head.weight @ x
            return x.float(), state

##########################################################################################################
# Step 1: set model & config (use v4 to run your trained-from-scratch models. v4 and v4neo are compatible)
##########################################################################################################

from transformers import PreTrainedTokenizerFast
TOKENIZER = PreTrainedTokenizerFast(tokenizer_file="20B_tokenizer.json")

args = types.SimpleNamespace()
args.RUN_DEVICE = "cpu" # 'cuda' // 'cpu' (already fast)
os.environ["TOKENIZERS_PARALLELISM"] = "false" # huggingface tokenizer setting to avoid deadlocks

# wget https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth
args.MODEL_NAME = "RWKV-4-Pile-169M-20220807-8023"
args.n_layer = 12
args.n_embd = 768
args.ctx_len = 1024
args.vocab_size = 50277

########################################################################################################
# Step 2: set prompt & sampling stuffs
########################################################################################################

context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, " + \
    "previously unexplored valley, in Tibet. Even more surprising to the researchers " + \
    "was the fact that the dragons spoke perfect Chinese."

NUM_TRIALS = 3
LENGTH_PER_TRIAL = 120

TEMPERATURE = 1.0
top_p = 0.8

print(f'\nUsing {args.RUN_DEVICE.upper()}. Loading {args.MODEL_NAME}...')

torch.set_float32_matmul_precision('high')
model = RWKV_RNN(args)

def sample_logits(out, temperature=1.0, top_p_usual=0.8):
    probs = F.softmax(out, dim=-1).cpu().numpy()
    sorted_probs = np.sort(probs)
    cumulative_probs = np.cumsum(sorted_probs) # [1,2,3] => [1,3,6]
    idx = np.argmax(cumulative_probs > top_p_usual) # vì là mảng True, False nên trả về idx của True đầu tiên
    cutoff = float(sorted_probs[idx]) # cutoff là tổng những prob lớn nhất đầu tiên vượt qua top_p_usual
    probs[probs < cutoff] = 0 # bỏ đi những prob < cutoff
    if temperature != 1.0: probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs) # chuẩn hóa lại probs sao cho tổng = 1
    return np.random.choice(a=len(probs), p=probs) # lấy mẫu

########################################################################################################
# Step 3: generate more tokens given the prompt
########################################################################################################

src_ctx = TOKENIZER.encode(context) # context => token_ids
src_len = len(src_ctx)
print(f"\nYour prompt has {src_len} tokens.")

init_state = None
next_token_id = src_ctx[-1]

# Khởi tạo init_state bằng cách chạy mô hình hết gần hết
for i in range(0, src_len): init_state = model.forward(src_ctx[i], init_state, True)

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ trial {TRIAL} ]-----------------\n', context, end="")

    state = init_state.clone()
    for i in range(src_len, src_len + LENGTH_PER_TRIAL): # sinh thêm LENGTH_PER_TRIAL
        out, state = model.forward(next_token_id, state)
        out[0] = -999999999  # disable <|endoftext|>
        next_token_id = sample_logits(out, TEMPERATURE, top_p) # lấy mẫu ngẫu nhiên token tiếp theo

        token = TOKENIZER.decode(next_token_id)
        if '\ufffd' not in token: # is valid utf8 string? (bỏ qua invalid token)
            print(token, end="", flush=True)
