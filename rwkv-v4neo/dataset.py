########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from utils import MaybeIsPrime


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        if args.data_type == "dummy":
            rank_zero_info("Building dummy data...")
            self.data = ""
            for i in range(100000):
                aa = (i) % 10000
                bb = (i * i) % 10000
                cc = aa + bb
                self.data += f".{aa}+{bb}={cc}."
        else:
            self.data = open(args.data_file, "r", encoding=args.data_type).read()

            rank_zero_info("Building token list...")
            unique = sorted(list(set(self.data)))
            self.vocab_size = len(unique)
            xx = 0
            xxObj = {}
            for u in unique:
                xxObj[xx] = u
                xx += 1
            with open(f"{args.proj_dir}/vocab.json", "w", encoding="utf-8") as vocab_file:
                vocab_file.write(json.dumps(xxObj, ensure_ascii=False))
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens, {self.vocab_size} vocab size.")
            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}


    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz


    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        if args.data_type == "uint16":
            i = np.random.randint(0, self.data_size-1)
            dix = self.data[i]
            x = torch.tensor(dix[:-1], dtype=torch.long)
            y = torch.tensor(dix[1:], dtype=torch.long)

        else:
            ctx_len = args.ctx_len
            req_len = ctx_len + 1
            magic_prime = args.magic_prime
            data = self.data

            if args.my_pile_stage > 0:
                ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank
                factor = (math.sqrt(5) - 1) / 2
                factor = int(magic_prime * factor)
                i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
                if data == self.data_pile:
                    i = i + args.my_pile_shift
                # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")
            else:
                # cheat: pick a random spot in dataset
                i = np.random.randint(0, self.data_size - req_len)

            if args.data_type == "binidx":
                dix = data.get(idx=0, offset=i, length=req_len).astype(int)
            # 
            elif args.data_type == "numpy":
                dix = data[i : i + req_len]
            # 
            else:
                dix = [self.stoi[s] for s in data[i : i + req_len]]

            x = torch.tensor(dix[:-1], dtype=torch.long)
            y = torch.tensor(dix[1:], dtype=torch.long)
            
        return x, y
