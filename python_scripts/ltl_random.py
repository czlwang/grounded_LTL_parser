#!/usr/bin/env python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from dataset import TextCollator, ParseDataset
from utils import *
from rewards import *
from model import make_parser, make_autoencoder
from dataset import infix2postfix

import statistics
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import yaml
import sys
import pathlib
import random

torch.autograd.set_detect_anomaly(True)

config = sys.argv[1]
with open(config) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

ordinal = cfg["cuda"]
USE_CUDA = torch.cuda.is_available()
DEVICE= torch.device(f'cuda:{ordinal}') if USE_CUDA else torch.device('cpu')
print(DEVICE)

OUT_DIR = cfg["out_dir"]
writer = SummaryWriter(f'runs/{OUT_DIR}')
print(f'Writing to {OUT_DIR}')

pathlib.Path(f'{OUT_DIR}').mkdir(parents=True, exist_ok=True) 

with open(f'{OUT_DIR}/config.txt', "w") as f:
    for k, v in cfg.items():
        f.write(f'\n{k} : {v}')

def gen_random_seq(n, trg_field):
    while True:
        words = [trg_field.vocab.sos_idx]
        length = 0
        for i in range(n):
            valids = trg_field.get_valid_idx(torch.tensor([words]))
            words.append(random.choice(valids[0]))
            length = i + 1
            if trg_field.vocab.eos_idx in words:
                break
        if trg_field.vocab.eos_idx in words:
            break
    pad_words = [trg_field.vocab.pad_idx]*(n)
    for i in range(len(words)-1):
        pad_words[i] = words[i+1]
    return pad_words, length

def get_random(n, max_len, trg_field):
    res = []
    lengths = []
    for i in range(n):
        rand_seq, rand_len = gen_random_seq(max_len, trg_field)
        res.append(rand_seq)
        lengths.append(rand_len)
    return res, lengths

def get_rand_metrics(example_iter, trg_field, ltl_args=None, args=None):
    #print("ltl metrics printing")
    f1_sum, fsa_sum, accept, count, executes = 0, 0, 0, 0, 0
    nl_trg, nl_trg_lengths = None, None
    for batch_idx, batch in enumerate(example_iter):
        mr_words, mr_lengths = get_random(batch["mr_trg_y"].shape[0], 15, trg_field)

        mr_words, mr_lengths = torch.tensor(mr_words), torch.tensor(mr_lengths)
        #mr_words, mr_lengths = batch["mr_trg_y"], batch["mr_trg_lengths"]#TODO

        mr_words = mr_words.unsqueeze(1)
        mr_lengths = mr_lengths.unsqueeze(1)
        f1_sum += string_match(mr_words, batch["mr_trg_y"], mr_lengths,
                               batch["mr_trg_lengths"], trg_field).sum()
        fsa_sum += fsa_match(mr_words, batch["mr_trg_y"], trg_field)
        accept += acceptance(mr_words, batch["ltl_data"], ltl_args, trg_field)
        count += mr_words.shape[0]
        executes += execute(mr_words, batch["mr_trg_y"], 
                            batch["ltl_data"], ltl_args, trg_field, args)
    f1 = f1_sum/count
    f1 = f1.item()
    fsa_score = fsa_sum/count
    accept = accept/count
    executes = executes/count
    return f1, fsa_score, accept, executes

def validate_sequences(seqs, trg_field):
    bad = []
    for seq in seqs:
        postfix = infix2postfix(seq, trg_field)
        #postfix = "workbench G !"
        #print()
        #print("testing", seq)
        #print(postfix)
        words = [trg_field.vocab.sos_idx]
        post_words = postfix.split(' ')
        post_words = [trg_field.vocab.stoi[s] for s in post_words]
        all_correct = True
        #print(post_words)
        for i in range(len(post_words)):
        #    print()
            valids = trg_field.get_valid_idx(torch.tensor([words]))[0]
        #    print("prev:", trg_field.get_string(words))
        #    print("next:", trg_field.vocab.itos[post_words[i]])
        #    print("valids:", [trg_field.vocab.itos[x] for x in valids])
            all_correct = post_words[i] in valids and all_correct
        #    print(all_correct)
            words.append(post_words[i])
        #print()
        valids = trg_field.get_valid_idx(torch.tensor([words]))[0]
        #print("prev:", trg_field.get_string(words))
        #print("next:", trg_field.vocab.itos[post_words[i]])
        #print("valids:", [trg_field.vocab.itos[x] for x in valids])

        all_correct = trg_field.vocab.eos_idx in valids and all_correct
        if not all_correct:
            print(postfix)
            bad.append(seq)
    for b in bad:
        print(b)
    print(len(bad))
    return all_correct

if __name__=="__main__":
    with open(cfg["data_dir"] + "train.src") as f:
        src_lines = f.readlines()
    with open(cfg["data_dir"] + "train.trg") as f:
        trg_lines = f.readlines()
    with open(cfg["data_dir"] + "train_ltl_data.json") as f:
        ltl_json = json.load(f)

    with open(cfg["data_dir"] + "test.src") as f:
        test_src_lines = f.readlines()
    with open(cfg["data_dir"] + "test.trg") as f:
        test_trg_lines = f.readlines()
    with open(cfg["data_dir"] + "test_ltl_data.json") as f:
        test_ltl_json = json.load(f)

    if cfg["sort"]:
        src_trg = list(zip(src_lines, trg_lines, ltl_json["data"]))
        src_trg = sorted(src_trg, key=lambda x: len(x[0]))
        src_lines, trg_lines, ltl_json["data"] = zip(*src_trg)

    src_field = Field("src", unk_token="<unk>", pad_token="<pad>", 
                            sos_token="<s>", eos_token="</s>")
    trg_field = Field("trg", unk_token="<unk>", pad_token="<pad>", 
                            sos_token="<s>", eos_token="</s>")

    src_field.build_vocab(src_lines)
    trg_field.build_vocab(trg_lines, trg=True, ltl=True)
    my_data = {}

    split_sizes = [cfg["train_split"], cfg["val_split"]]
    split_range = zip([sum(split_sizes[:i]) for i in range(len(split_sizes))],
                      [sum(split_sizes[:i+1]) for i in range(len(split_sizes))])
    splits = ["train", "val"]
    train_src_lines = []

    for split, part in zip(splits, split_range):
        split_ltl = {"data": ltl_json["data"][part[0]:part[1]]}
        my_data[split] = ParseDataset(src_lines[part[0]:part[1]], 
                                      trg_lines[part[0]:part[1]],
                                      src_field, trg_field,
                                      ltl_json=split_ltl,
                                      ltl=True)
        if split == "train":
           train_src_lines = src_lines[part[0]:part[1]] 
        with open(f'{OUT_DIR}/{split}.src', 'w') as f:
            f.write('\n'.join(src_lines[part[0]:part[1]]))
        with open(f'{OUT_DIR}/{split}.trg', 'w') as f:
            f.write('\n'.join(trg_lines[part[0]:part[1]]))

    my_data["test"] = ParseDataset(test_src_lines, 
                                   test_trg_lines,
                                   src_field, trg_field,
                                   ltl_json=test_ltl_json,
                                   ltl=True)

    my_collator = TextCollator(pad_idx=1, device='cpu')

    my_loaders = {}
    for split in ["train", "val", "test"]:
        shuffle=False
        if split in ["train"]:
            batch_size = cfg["train_batch_size"]
            #shuffle = True
        else:
            batch_size = cfg["val_batch_size"]
        my_loaders[split] = DataLoader(my_data[split], 
                                       batch_size=batch_size,
                                       collate_fn=my_collator,
                                       shuffle=shuffle)

    #print_data_info(my_data, src_field, trg_field)

    #validate_sequences(test_trg_lines, trg_field)
    #exit()  
    ltl_args = get_args()
    f1_sum, equiv_sum, accept_sum, exec_sum = [], [], [], []
    for i in range(5):
        args = cfg.copy()
        f1, equiv, accept, exe = get_rand_metrics(my_loaders["test"], 
                                                      trg_field, ltl_args, args)
        f1_sum.append(f1)
        equiv_sum.append(equiv)
        accept_sum.append(accept)
        exec_sum.append(exe)
    print("f1", statistics.mean(f1_sum), statistics.stdev(f1_sum))
    print("equiv", statistics.mean(equiv_sum), statistics.stdev(equiv_sum))
    print("accept", statistics.mean(accept_sum), statistics.stdev(accept_sum))
    print("execute", statistics.mean(exec_sum), statistics.stdev(exec_sum))
