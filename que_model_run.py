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
args.MODEL_NAME = "model/rwkv-26.pth"
args.n_layer = 6
args.n_embd = 512

# Khởi tạo char tokenizer, ITOC: token_id => char, CTOI: char to token_id
ITOC = {0: "\n", 1: " ", 2: "\"", 3: "%", 4: "&", 5: "'", 6: "(", 7: ")", 8: "*", 9: "+", 10: ",", 11: "-", 12: ".", 13: "/", 14: 0, 15: 1, 16: 2, 17: 3, 18: 4, 19: 5, 20: 6, 21: 7, 22: 8, 23: 9, 24: ":", 25: ";", 26: "<", 27: "=", 28: "A", 29: "B", 30: "C", 31: "D", 32: "E", 33: "F", 34: "G", 35: "H", 36: "I", 37: "J", 38: "K", 39: "L", 40: "M", 41: "N", 42: "O", 43: "P", 44: "Q", 45: "R", 46: "S", 47: "T", 48: "U", 49: "V", 50: "W", 51: "X", 52: "Y", 53: "Z", 54: "[", 55: "]", 56: "_", 57: "a", 58: "b", 59: "c", 60: "d", 61: "e", 62: "f", 63: "g", 64: "h", 65: "i", 66: "j", 67: "k", 68: "l", 69: "m", 70: "n", 71: "o", 72: "p", 73: "q", 74: "r", 75: "s", 76: "t", 77: "u", 78: "v", 79: "w", 80: "x", 81: "y", 82: "z", 83: "¬", 84: "­", 85: "±", 86: "À", 87: "Á", 88: "Â", 89: "Ã", 90: "É", 91: "Ê", 92: "Ì", 93: "Í", 94: "Ð", 95: "Ò", 96: "Ó", 97: "Ô", 98: "Õ", 99: "Ù", 100: "Ú", 101: "Ý", 102: "à", 103: "á", 104: "â", 105: "ã", 106: "è", 107: "é", 108: "ê", 109: "ì", 110: "í", 111: "ò", 112: "ó", 113: "ô", 114: "õ", 115: "ù", 116: "ú", 117: "ý", 118: "Ă", 119: "ă", 120: "Đ", 121: "đ", 122: "Ĩ", 123: "ĩ", 124: "Ũ", 125: "ũ", 126: "Ơ", 127: "ơ", 128: "Ư", 129: "ư", 130: "̀", 131: "́", 132: "̃", 133: "̉", 134: "̣", 135: "α", 136: "β", 137: "Ạ", 138: "ạ", 139: "Ả", 140: "ả", 141: "Ấ", 142: "ấ", 143: "Ầ", 144: "ầ", 145: "Ẩ", 146: "ẩ", 147: "Ẫ", 148: "ẫ", 149: "Ậ", 150: "ậ", 151: "Ắ", 152: "ắ", 153: "Ằ", 154: "ằ", 155: "Ẳ", 156: "ẳ", 157: "Ẵ", 158: "ẵ", 159: "Ặ", 160: "ặ", 161: "Ẹ", 162: "ẹ", 163: "Ẻ", 164: "ẻ", 165: "ẽ", 166: "Ế", 167: "ế", 168: "Ề", 169: "ề", 170: "Ể", 171: "ể", 172: "Ễ", 173: "ễ", 174: "Ệ", 175: "ệ", 176: "Ỉ", 177: "ỉ", 178: "Ị", 179: "ị", 180: "Ọ", 181: "ọ", 182: "Ỏ", 183: "ỏ", 184: "Ố", 185: "ố", 186: "Ồ", 187: "ồ", 188: "Ổ", 189: "ổ", 190: "Ỗ", 191: "ỗ", 192: "Ộ", 193: "ộ", 194: "Ớ", 195: "ớ", 196: "Ờ", 197: "ờ", 198: "Ở", 199: "ở", 200: "Ỡ", 201: "ỡ", 202: "Ợ", 203: "ợ", 204: "Ụ", 205: "ụ", 206: "Ủ", 207: "ủ", 208: "Ứ", 209: "ứ", 210: "Ừ", 211: "ừ", 212: "Ử", 213: "ử", 214: "Ữ", 215: "ữ", 216: "Ự", 217: "ự", 218: "Ỳ", 219: "ỳ", 220: "Ỷ", 221: "ỷ", 222: "Ỹ", 223: "ỹ", 224: "‎", 225: "–", 226: "’", 227: "“", 228: "”", 229: "•"}
CTOI = {}
for k, v in ITOC.items(): CTOI[v] = k
def encode(str): return [CTOI[char] for char in str] # => token_ids
def decode(token_id): return ITOC[token_id] # => char

########################################################################################################
# Step 2: set prompt & sampling stuffs
########################################################################################################

context = random.choice([
    "abc",
    # "nghĩa vụ nộp thuế", 
    # "quyền lợi công dân", 
    # "bộ luật dân sự",
]) + " "

NUM_TRIALS = 3
LENGTH_PER_TRIAL = 250
TEMPERATURE = 0.9
top_p = 0.8

def sample_logits(out, temperature=1.0, top_p_usual=0.8):
    probs = F.softmax(out, dim=-1).numpy()
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

# Nhồi context (a.k.a prompt) vào mô hình
for token_id in encode(context):
    out, init_state = model.forward(token_id, init_state)

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Lần thử {TRIAL} ]-----------------')
    print(f'Hỏi: {context}\nĐáp: {context}', end="")
    state = init_state.clone() # clone() để giữ nguyên init_state cho lần TRIAL tiếp theo
    for i in range(LENGTH_PER_TRIAL): # sinh thêm LENGTH_PER_TRIAL tokens nữa từ prompt đầu vào
        token_id = sample_logits(out, TEMPERATURE, top_p) # lấy mẫu ngẫu nhiên token tiếp theo
        print(decode(token_id), end="", flush=True)
        out, state = model.forward(token_id, state)
