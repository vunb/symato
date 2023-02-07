import numpy as np
import random, types, torch
from torch.nn import functional as F

########################################################################################################
# Step 0: Define the model in inference mode
########################################################################################################

class RWKV_RNN(torch.jit.ScriptModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval() # set torch to inference mode
        
        # Load tham số từ file vào vào bộ nhớ và biến đổi cho phù hợp
        w = torch.load(args.MODEL_NAME, map_location='cpu')
        for k in w.keys():
            if      '.time_' in k: w[k] = w[k].squeeze() # (A,1,B,1) => (A,B)
            if '.time_decay' in k: w[k] = -torch.exp(w[k].float()) # e^negative = decay it's actually `e^{-e^x}``
            else: w[k] = w[k].float() # convert to f32 type

        # Gán tham số vào biến số mô hình
        self.w = types.SimpleNamespace()
        self.w.blocks = {}
        for k in w.keys():
            print(k, w[k].shape)
            parts = k.split('.') # Ví dụ k = "blocks.0.att.value.weight" => parts = ['block','0','ln0','weight']
            last = parts.pop() # => last = "weight"; parts = ['block','0','ln0']
            here = self.w
            for p in parts: # từng bước mở rộng namespace
                if p.isdigit(): # tầng thứ p
                    p = int(p) # dùng [] vì here (w.blocks) là dict object {}
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else: # dùng hasattr, setattr, getattr vì here là types.SimpleNamespace()
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k]) # gán giá trị vào namespace cuối cùng => self.w.blocks[0].ln0.weight = w[k]


    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @torch.jit.script_method
    def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        ffn_xx = 5*i+0 # feed-forward or channel mixing
        # token-shift with diff mixing factors for k and r
        xk = x * time_mix_k + state[ffn_xx] * (1 - time_mix_k)
        xr = x * time_mix_r + state[ffn_xx] * (1 - time_mix_r)
        state[ffn_xx] = x # prev_x = x

        r = torch.sigmoid(rw @ xr) # receptance factor thuộc 0 -> 1
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        return r * (vw @ k)

    @torch.jit.script_method
    def time_mixing(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        att_xx = 5*i+1 # attention or time mixing
        # token-shift with diff mixing factors for k, v and r
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

    def forward(self, token_id, state, preprocess_only=False):
        with torch.no_grad():
            # 0/ Khởi tạo trạng thái hệ thống nếu chưa được khởi tạo
            if state == None:
                state = torch.zeros(self.args.n_layer * 5, self.args.n_embd)
                for i in range(self.args.n_layer): state[5*i+4] = -1e30 # state[att_pp] = âm vô cực

            # 1/ Lấy vector nhúng của token_id
            x = self.w.emb.weight[token_id]
            # Và áp dụng layer-norm-0 ở tầng đầu tiên để small-init-emb trick hoạt động
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            
            # 2/ Với mỗi tầng áp dụng:
            for i in range(self.args.n_layer):
                # 2.1/ time-mixing
                att = self.w.blocks[i].att # trọng số của khối time-mixing
                x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i, 
                    att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay, 
                    att.key.weight, att.value.weight, att.receptance.weight, att.output.weight)

                # 2.2/ channel-mixing
                ffn = self.w.blocks[i].ffn # trọng số của khối channel-mixing
                x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i, 
                    ffn.time_mix_k, ffn.time_mix_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)

            if preprocess_only: return state
            # 3/ Cuối cùng áp dụng bộ phân lớp cho ra next token probabilities
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state

##########################################################################################################
# Step 1: set model & config (use v4 to run your trained-from-scratch models. v4 and v4neo are compatible)
##########################################################################################################
args = types.SimpleNamespace()
args.MODEL_NAME = "rwkv-v4neo/out/rwkv-0.pth"
args.n_layer = 6
args.n_embd = 512

# Khởi tạo char tokenizer, ITOC: token_id => char, CTOI: char to token_id
ITOC = {0: "\t", 1: "\n", 2: "\u0010", 3: " ", 4: "!", 5: "\"", 6: "#", 7: "$", 8: "%", 9: "&", 10: "'", 11: "(", 12: ")", 13: "*", 14: "+", 15: ",", 16: "-", 17: ".", 18: "/", 19: 0, 20: 1, 21: 2, 22: 3, 23: 4, 24: 5, 25: 6, 26: 7, 27: 8, 28: 9, 29: ":", 30: ";", 31: "<", 32: "=", 33: ">", 34: "?", 35: "@", 36: "A", 37: "B", 38: "C", 39: "D", 40: "E", 41: "F", 42: "G", 43: "H", 44: "I", 45: "J", 46: "K", 47: "L", 48: "M", 49: "N", 50: "O", 51: "P", 52: "Q", 53: "R", 54: "S", 55: "T", 56: "U", 57: "V", 58: "W", 59: "X", 60: "Y", 61: "Z", 62: "[", 63: "]", 64: "_", 65: "`", 66: "a", 67: "b", 68: "c", 69: "d", 70: "e", 71: "f", 72: "g", 73: "h", 74: "i", 75: "j", 76: "k", 77: "l", 78: "m", 79: "n", 80: "o", 81: "p", 82: "q", 83: "r", 84: "s", 85: "t", 86: "u", 87: "v", 88: "w", 89: "x", 90: "y", 91: "z", 92: "~", 93: " ", 94: "­", 95: "·", 96: "À", 97: "Á", 98: "Â", 99: "Ã", 100: "Ê", 101: "Ì", 102: "Í", 103: "Ò", 104: "Ó", 105: "Ô", 106: "Ù", 107: "Ú", 108: "Ý", 109: "à", 110: "á", 111: "â", 112: "ã", 113: "è", 114: "é", 115: "ê", 116: "ì", 117: "í", 118: "ò", 119: "ó", 120: "ô", 121: "õ", 122: "ù", 123: "ú", 124: "ý", 125: "Ă", 126: "ă", 127: "Đ", 128: "đ", 129: "ĩ", 130: "ũ", 131: "Ơ", 132: "ơ", 133: "Ư", 134: "ư", 135: "̀", 136: "́", 137: "̂", 138: "̃", 139: "̆", 140: "̉", 141: "̛", 142: "̣", 143: "е", 144: "ѕ", 145: "і", 146: "Ạ", 147: "ạ", 148: "Ả", 149: "ả", 150: "Ấ", 151: "ấ", 152: "Ầ", 153: "ầ", 154: "Ẩ", 155: "ẩ", 156: "Ẫ", 157: "ẫ", 158: "Ậ", 159: "ậ", 160: "ắ", 161: "ằ", 162: "ẳ", 163: "ẵ", 164: "ặ", 165: "ẹ", 166: "ẻ", 167: "ẽ", 168: "Ế", 169: "ế", 170: "Ề", 171: "ề", 172: "Ể", 173: "ể", 174: "Ễ", 175: "ễ", 176: "Ệ", 177: "ệ", 178: "ỉ", 179: "Ị", 180: "ị", 181: "ọ", 182: "ỏ", 183: "Ố", 184: "ố", 185: "ồ", 186: "Ổ", 187: "ổ", 188: "Ỗ", 189: "ỗ", 190: "Ộ", 191: "ộ", 192: "Ớ", 193: "ớ", 194: "ờ", 195: "Ở", 196: "ở", 197: "ỡ", 198: "Ợ", 199: "ợ", 200: "Ụ", 201: "ụ", 202: "Ủ", 203: "ủ", 204: "Ứ", 205: "ứ", 206: "ừ", 207: "Ử", 208: "ử", 209: "Ữ", 210: "ữ", 211: "Ự", 212: "ự", 213: "Ỳ", 214: "ỳ", 215: "ỷ", 216: "ỹ", 217: " ", 218: "‎", 219: "–", 220: "‘", 221: "’", 222: "“", 223: "”", 224: "•", 225: "…", 226: "≤", 227: "≥", 228: "⋅", 229: "❤", 230: ""}
CTOI = {}
for k, v in ITOC.items(): CTOI[v] = k
def encode(str): return [CTOI[char] for char in str] # => token_ids
def decode(token_id): return ITOC[token_id] # => char

########################################################################################################
# Step 2: set prompt & sampling stuffs
########################################################################################################

NUM_TRIALS = 3
LENGTH_PER_TRIAL = 1000
TEMPERATURE = 0.9
top_p = 0.8
STOP_TID = 2

def sample_logits(out, temperature=1.0, top_p_usual=0.8):
    probs = F.softmax(out, dim=-1).numpy()
    if probs.argmax() == STOP_TID: return STOP_TID
    sorted_probs = np.sort(probs)
    cumulative_probs = np.cumsum(sorted_probs) # [1,2,3] => [1,3,6]
    idx = np.argmax(cumulative_probs > top_p_usual) # vì là mảng True, False nên trả về idx của True đầu tiên
    cutoff = float(sorted_probs[idx]) # cutoff là tổng những prob lớn nhất đầu tiên vượt qua top_p_usual
    probs[probs < cutoff] = 0 # bỏ đi những prob < cutoff
    if temperature != 1.0: probs = np.power(probs, 1.0 / temperature)
    probs = probs / np.sum(probs) # chuẩn hóa lại probs sao cho tổng = 1
    return np.random.choice(a=len(probs), p=probs) # lấy mẫu

########################################################################################################
# Step 3: generate more tokens given the prompt
########################################################################################################

print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
model = RWKV_RNN(args)
out, init_state = None, None

question = random.choice([
    # "chuyển đổi số là gì", 
    # "Cách thực hiện đúng của chuyển đổi số là gì",
    "thẻ nhà báo hết hạn xử như thế nào",
]) + "?"

context = f"\nQA_BEGIN\nCâu hỏi: {question}\nTrả lời: "
# Nhồi context (a.k.a prompt) vào mô hình
for token_id in encode(context):
    out, init_state = model.forward(token_id, init_state)

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Lần thử {TRIAL} ]-----------------')
    print(f'Hỏi: {question}\nĐáp: ', end="")
    state = init_state.clone() # clone() để giữ nguyên init_state cho lần TRIAL tiếp theo
    for i in range(LENGTH_PER_TRIAL): # sinh thêm LENGTH_PER_TRIAL tokens nữa từ prompt đầu vào
        token_id = sample_logits(out, TEMPERATURE, top_p)
        if token_id == STOP_TID: break
        print(decode(token_id), end="", flush=True)
        out, state = model.forward(token_id, state)
