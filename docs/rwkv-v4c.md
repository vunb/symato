# Cải tiến

## Full self-attn

```py
@MyFunction
def self_attn(self, q, k, v):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.att_mask == 0, float('-inf'))
    att = F.softmax(att, dim = -1)
    x = att @ v
    return x

@MyFunction
def token_shift_and_projection(self, x):
    # Mix x with the previous timestep to produce xk, xv, xr
    xx = self.time_shift(x)
    xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
    xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
    xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
    xqq = x * self.time_mix_qq + xx * (1 - self.time_mix_qq)
    xkk = x * self.time_mix_kk + xx * (1 - self.time_mix_kk)
    xvv = x * self.time_mix_vv + xx * (1 - self.time_mix_vv)

    # Use xk, xv, xr to produce k, v, r
    k = self.key(xk)
    v = self.value(xv)
    r = self.receptance(xr)
    sr = torch.sigmoid(r)
    
    qq = self.qq(xqq)
    kk = self.kk(xkk)
    vv = self.vv(xvv)

    return sr, k, v, qq, kk, vv

def forward(self, x):
    B, T, C = x.size()  # x = (Batch,Time,Channel)
    sr, k, v, qq, kk, vv = self.token_shift_and_projection(x)
    rwkv = sr * RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v)
    rwkv = self.output(rwkv) + self.oo(self.self_attn(qq, kk, vv))
    return rwkv # là phép cộng của rwkv time-mixing (thông thường) với self-attn (thông thường)
```

## Thay channel-mixing bằng MishGLU
```py
class MishGLU(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = self.args.my_testing
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)
```
