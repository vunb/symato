########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        if args.data_type == "symato":
            import sys; sys.path.append('../')
            from symato import Symato
            smt = Symato()
            self.data  = smt.tokenize(args.data_file)
            self.vocab_size = smt.vocab_size()
            self.data_size = len(self.data)
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            rank_zero_info(f"Data has {self.data_size} samples.")

        else: # unicode
            txt = open(args.data_file, "r", encoding=args.data_type).read()
            from tokenization_phobert_fast import PhobertTokenizerFast
            os.environ["TOKENIZERS_PARALLELISM"] = "False"
            tknz = PhobertTokenizerFast("./data/vocab.txt", "./data/bpe.codes", "./data/tokenizer.json")
            self.vocab_size = 64256 # 251 * 256
            self.data = tknz.encode(txt)
            self.data_size = len(self.data)
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            rank_zero_info(f"Data has {self.data_size} samples.")

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz


    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size # total number of training processes

        ctx_len = args.ctx_len # ctx_len là độ dài chuỗi token đầu vào
        req_len = ctx_len + 1  # cộng thêm một token là kết quả đầu ra 
        magic_prime = args.magic_prime
        data = self.data
        i = np.random.randint(0, self.data_size - req_len)
        dix = data[i : i + req_len]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
            
        return x, y
