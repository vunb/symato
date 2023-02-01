https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/src/model.py#L258

```py
# def __init__(self, args, layer_id):
self.args = args
self.layer_id = layer_id

self.ln1 = nn.LayerNorm(args.n_embd)
self.ln2 = nn.LayerNorm(args.n_embd)

if self.layer_id == 0:
    self.ln0 = nn.LayerNorm(args.n_embd)
    if args.my_pos_emb > 0:
        self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
        self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

if self.layer_id == 0 and self.args.pre_ffn > 0:
    self.ffnPre = RWKV_ChannelMix(args, 0)
else:
    self.att = RWKV_TimeMix(args, layer_id)

self.ffn = RWKV_ChannelMix(args, layer_id)

self.tiny_ln = nn.LayerNorm(args.n_embd)
self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

# def forward(self, x, x_emb=None):
B, T, C = x.size()
if self.layer_id == 0:
    x = self.ln0(x)
    if args.my_pos_emb > 0: # learnable pos embeding (nhiều khả năng không dùng tới)
        pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
        x = x + pos_emb

if self.layer_id == 0 and args.pre_ffn > 0:
    x = x + self.ffnPre(self.ln1(x))
else:
    x = x + self.att(self.ln1(x))
x = x + self.ffn(self.ln2(x))

if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
    xx = self.tiny_ln(x)
    q = self.tiny_q(xx)[:, :T, :]
    k = self.tiny_k(xx)[:, :T, :]
    c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
    c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
    x = x + c @ self.tiny_v(x_emb)
return x


# class RWKV(pl.LightningModule):
self.emb = nn.Embedding(args.vocab_size, args.n_embd)
self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

self.ln_out = nn.LayerNorm(args.n_embd)
self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

 # def forward(self, idx):
B, T = idx.size()
assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
x = self.emb(idx)
x_emb = x
x = block(x, x_emb)
x = self.ln_out(x)
x = self.head(x)
return x
```

Tiny attn
```py
self.tiny_ln = nn.LayerNorm(args.n_embd)
self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
self.register_buffer("casual_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

B, T, C = x.size()
xx = self.tiny_ln(x)
q = self.tiny_q(xx)[:, :T, :]
k = self.tiny_k(xx)[:, :T, :]
att = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
att = att.masked_fill(self.casual_mask[:T, :T] == 0, 0) # att[i,j]=0 where j > i
x = x + att @ self.tiny_v(x_emb)
```
=> __tiny_attn không dùng softmax__