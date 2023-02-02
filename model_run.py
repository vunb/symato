########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
# https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/src/model_run.py
########################################################################################################

# tham khảo https://github.com/harrisonvanderbyl/rwkvstic/blob/master/src/rwkvstic/agnostic/agnosticRwkv.py
# hazardous1222: removing all tensor pre-prep and I assume some of the Math Blink used to try and make fp16 work

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

RWKV_HEAD_QK_DIM = 0
RWKV_RESCALE_LAYER = 6
# set x=x/2 every 6 block to prevent FP16 overflow. Peng Bo scale some weight too, so the final result is the same
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM} RWKV_JIT_ON {RWKV_JIT_ON} RWKV_RESCALE_LAYER {RWKV_RESCALE_LAYER}\n')

############################################################################################################

class RWKV_RNN(MyModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.FLOAT_MODE = args.FLOAT_MODE
        self.RUN_DEVICE = args.RUN_DEVICE

        with torch.no_grad():
            # Example: w = torch.load('RWKV-4a-Pile-170M-20221209-7955.pth', map_location='cpu')
            w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
            # refine weights and send to correct device
            for k in w.keys():
                block_id = int(k.split('.')[1]) if 'blocks.' in k else 0 # xác định block_id
                rescale = 2 ** int(block_id // RWKV_RESCALE_LAYER) # set x=x/2 every 6 block
                if 'att.output.weight' in k: w[k] = w[k] / rescale
                if  'ffn.value.weight' in k: w[k] = w[k] / rescale
                if      '.time_' in k: w[k] = w[k].squeeze() # (A,1,B,1) => (A,B)
                if '.time_decay' in k: w[k] = -torch.exp(w[k].float())#; print(">>> time_decay", w[k])
                # e ^ negative = decay it's actually e ^ ( - e^ x ) biến time_decay thành số âm
                if '.time_first' in k: w[k] = w[k].float()
                else: # các tham số khác
                    if   self.FLOAT_MODE == "fp32": w[k] = w[k].float()
                    elif self.FLOAT_MODE == "bf16": w[k] = w[k].bfloat16()
                    elif self.FLOAT_MODE == "fp16": w[k] = w[k].half()

                w[k].requires_grad = False # chỉ inference, nên không cần gradient
                if args.RUN_DEVICE == 'cuda' and k != 'emb.weight': 
                    w[k] = w[k].cuda() # emb lookup table stay in ram

                if ('blocks.' not in k) or ('blocks.0.' in k):
                    print(k.ljust(40), str(w[k].dtype).replace('torch.', '').ljust(10), w[k].device)
                else:
                    print('.', end = '', flush = True)

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

    def get_converted_prev_x_and_xx(self, prev_x, x):
        if self.FLOAT_MODE == "bf16": return prev_x.type(torch.bfloat16), x.float()
        if self.FLOAT_MODE == "fp16": return prev_x.half(), x.float()
        return prev_x, x

    # state[] i+0=ffn_xx i+1=att_xx i+2=att_aa i+3=att_bb i+4=att_pp
    @MyFunction
    def FF(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        ffn_xx = 5*i+0 # ffn = channel mixing
        prev_x, xx = self.get_converted_prev_x_and_xx(state[ffn_xx], x)

        # token-shift with diff mixing factors for k and r
        xk = x * time_mix_k + prev_x * (1 - time_mix_k)
        xr = x * time_mix_r + prev_x * (1 - time_mix_r)
        state[ffn_xx] = xx # prev_x = xx

        r = torch.sigmoid(rw @ xr) # receptance factor: 0 -> 1
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        kv = vw @ k
        return r * kv

    # state[] i+0=ffn_xx i+1=att_xx i+2=att_aa i+3=att_bb i+4=att_pp
    @MyFunction
    def SA(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        att_xx = 5*i+1 # attention or time mixing
        prev_x, xx = self.get_converted_prev_x_and_xx(state[att_xx], x)

        # token-shift
        xk = x * time_mix_k + prev_x * (1 - time_mix_k)
        xv = x * time_mix_v + prev_x * (1 - time_mix_v)
        xr = x * time_mix_r + prev_x * (1 - time_mix_r)
        state[att_xx] = xx # prev_x = xx

        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv

        if '16' in self.FLOAT_MODE:
            kk = k.float()
            vv = v.float()
        else:
            kk = k
            vv = v

        aa = state[5*i+2] # exponential moving average of kv
        bb = state[5*i+3] # exponential moving average of k
        pp = state[5*i+4] # idea: use pp to store exponent of a and b

        ww = time_first + kk # u + k_i
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2

        ww = pp + time_decay
        qq = torch.maximum(ww, kk)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(kk - qq)
        state[5*i+2] = e1 * aa + e2 * vv
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = qq

        wkv = a / b
        if   self.FLOAT_MODE == "bf16": wkv = wkv.type(torch.bfloat16)
        elif self.FLOAT_MODE == "fp16": wkv = wkv.half()
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
                if i == 0: x = self.LN(x, w.blocks[i].ln0) # áp dụng LN ở tầng đầu

                # time-mixing
                ww = w.blocks[i].att # trọng số att
                x = x + self.SA(self.LN(x, w.blocks[i].ln1), state, i, 
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                
                # channel-mixing
                ww = w.blocks[i].ffn # trọng số fnn
                x = x + self.FF(self.LN(x, w.blocks[i].ln2), state, i, 
                    ww.time_mix_k, ww.time_mix_r, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight)
                
                # rescale x = x / 2 for each 6 block to prevent FP16 overflow
                if (i+1) % RWKV_RESCALE_LAYER == 0: x = x / 2

            if preprocess_only: return state

            x = self.LN(x, w.ln_out)
            x = w.head.weight @ x
            return x.float(), state
