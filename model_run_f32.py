import types
import torch
import math, os, gc
from torch.nn import functional as F
import torch.nn as nn
from typing import List, Dict

MyModule = nn.Module
def __nop(ob): return ob
MyFunction = __nop

# try torch jit --> faster for fp32, slower for fp16 (why?)
RWKV_JIT_ON = os.environ.get("RWKV_JIT_ON", "1") # if not set, enable by default
if  RWKV_JIT_ON == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

############################################################################################################

class RWKV_RNN(MyModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.RUN_DEVICE = args.RUN_DEVICE

        with torch.no_grad():
            w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
            # refine weights and send to correct device
            for k in w.keys():
                block_id = int(k.split('.')[1]) if 'blocks.' in k else 0 # xác định block_id
                if      '.time_' in k: w[k] = w[k].squeeze() # (A,1,B,1) => (A,B)
                if '.time_decay' in k: w[k] = -torch.exp(w[k].float()) # e ^ negative = decay it's actually e ^ ( - e^ x )
                else: w[k] = w[k].float()

                w[k].requires_grad = False # chỉ inference, nên không cần gradient
                if args.RUN_DEVICE == 'cuda' and k != 'emb.weight':
                    w[k] = w[k].cuda() # emb lookup table stay in ram

        # store weights in self.w
        self.w = types.SimpleNamespace()
        for k in w.keys():
            parts = k.split('.') # blocks.0.att.value.weight => ['block','0','att','value','weight']
            last = parts.pop() # => last = weight; parts = ['block','0','att','value']
            here = self.w
            for i, p in enumerate(parts): # mở rộng namespace
                if p.isdigit():
                    p = int(p) # dùng [] vì here (w.blocks) là dict object {}
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else: # dùng hasattr, setattr, getattr vì here là types.SimpleNamespace()
                    if not hasattr(here, p):
                        if p == "blocks": setattr(here, p, {})
                        else: setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k]) # gán giá trị vào name part cuối cùng

        self.eval() # torch eval mode (not train mode)
        gc.collect() # giải phóng ram
        torch.cuda.empty_cache() # giải phóng vram

    def LN(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    # state[] i+0=ffn_xx i+1=att_xx i+2=att_aa i+3=att_bb i+4=att_pp
    @MyFunction
    def FF(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        ffn_xx = 5*i+0
        # token-shift with diff mixing factors for k and r
        xk = x * time_mix_k + state[ffn_xx] * (1 - time_mix_k)
        xr = x * time_mix_r + state[ffn_xx] * (1 - time_mix_r)
        state[ffn_xx] = x # prev_x = x

        r = torch.sigmoid(rw @ xr) # receptance factor thuộc 0 -> 1
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        return r * (vw @ k)

    # state[] i+0=ffn_xx i+1=att_xx i+2=att_aa i+3=att_bb i+4=att_pp
    @MyFunction
    def SA(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
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

    def forward(self, ctx, state, preprocess_only = False):
        with torch.no_grad():
            w = self.w
            args = self.args

            x = w.emb.weight[ctx[-1]]
            if self.RUN_DEVICE == 'cuda': x = x.cuda()

            if state == None: # khởi tạo trạng thái hệ thống
                state = torch.zeros(args.n_layer * 5, args.n_embd, device=self.RUN_DEVICE)
                for i in range(args.n_layer): state[5*i+4] -= 1e30 # att_pp = dương vô cực

            # Với mỗi tầng
            for i in range(args.n_layer):
                if i == 0: # áp dụng layer-norm-0 ở tầng đầu tiên để small-init-emb trick hoạt động
                    x = self.LN(x, w.blocks[i].ln0)

                # time-mixing
                att = w.blocks[i].att # trọng số của khối time-mixing
                x = x + self.SA(self.LN(x, w.blocks[i].ln1), state, i, 
                    att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay, 
                    att.key.weight, att.value.weight, att.receptance.weight, att.output.weight)
                
                # channel-mixing
                ffn = w.blocks[i].ffn # trọng số của khối channel-mixing
                x = x + self.FF(self.LN(x, w.blocks[i].ln2), state, i, 
                    ffn.time_mix_k, ffn.time_mix_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
                
            if preprocess_only: return state

            x = self.LN(x, w.ln_out)
            x = w.head.weight @ x
            return x.float(), state
