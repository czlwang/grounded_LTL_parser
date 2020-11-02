#!/usr/bin/env python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from dataset import ActionCollator, ActionDataset
from utils import *
from rewards import *
from model import make_parser, make_autoencoder, make_seq2seq_base

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
import pickle

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

def compute_q(beam_out, unmasked_beam_words, beta, reward):
    '''
        beam_out = [batch_size, beam_size, seq_len, vocab_size]
        unmasked_beam_words = [batch_size, beam_size, seq_len]
        beta = temperature
        reward = [batch_size, beam_size]
    '''
    q_log_probs = beam_out.gather(3, unmasked_beam_words).squeeze(-1)
    #q_log_probs [batch, beam, seq_len]
    q_probs = torch.exp(q_log_probs.sum(-1))#[batch, beam]
    q = q_probs*reward
    norm = q.sum(-1).unsqueeze(-1)#[batch]
    norm[norm==0] = 1 #prevent div by zero
    q_mml = q/norm

    q = torch.pow(q_mml, beta)
    q_norm = q.sum(-1).unsqueeze(-1)
    q_norm[q_norm==0] = 1
    q = q/q_norm
    q = q.detach()
    return q

def compute_nl_loss(unmasked_beam_words=None, beta=0.0,
                    nl_trg=None, loss_type=None,
                    mc_nl_words=None, mc_nl_out=None, 
                    nl_lengths=None):
        batch_size, beam_width, nl_seq_len, nl_vocab = mc_nl_out.shape
        nl_trg = nl_trg.unsqueeze(1).expand(-1, beam_width, -1)
        nl_lengths = nl_lengths.unsqueeze(-1).expand(-1, beam_width)

        mask_tmp = make_mask(nl_lengths, nl_trg)
        mc_nl_mask = mask_tmp.clone().to(DEVICE).float()
        del mask_tmp

        mc_nl_word_probs = mc_nl_out.gather(3, nl_trg.unsqueeze(-1)).squeeze(-1)
        #word_probs (before squeeze) is [batch_size, beam_size, seq_len, 1]

        mc_nl_word_probs = mc_nl_word_probs*mc_nl_mask

        mc_nl_word_probs = mc_nl_word_probs.sum(-1)#[batch, beam], p(x|y)
        nl_loss = mc_nl_word_probs.mean(-1).mean()
        #NOTE check beta
        nl_loss = -nl_loss
        return nl_loss

def compute_loss(beam_out, beam_words, tensor_lengths, reward, trg=None,
                 train_mode="rl", unmasked_beam_words=None, beta=0.0):
    '''
    inputs:
        beam_out = [batch_size, beam_size, seq_len, vocab_size]
        beam_words = [batch_size, beam_size, seq_len]
        tensor_lengths = [batch_size, beam_size, seq_len]: lengths of the beam words
        reward = [batch_size, beam_size]
        note beam_words seq_len is +1 beam_out seq_len (includes start token)
    '''
    batch_size, beam_size, seq_len = beam_words.shape
    beam_words = beam_words.unsqueeze(-1).to(DEVICE)
    unmasked_beam_words = unmasked_beam_words.unsqueeze(-1).to(beam_out.device)
    #beam_words is now [batch_size, beam_size, seq_len, 1]

    word_probs = beam_out.gather(3, unmasked_beam_words).squeeze(-1)
    #word_probs (before squeeze) is [batch_size, beam_size, seq_len, 1]

    beam_words = beam_words.squeeze()

    mask_tmp = make_mask(tensor_lengths, beam_words)
    mask = mask_tmp.clone().to(DEVICE).float()
    del mask_tmp

    #for computing MLE loss
    if trg is not None and train_mode=="ml":
        trg_repeat = trg.unsqueeze(1).repeat(1,beam_size,1).unsqueeze(-1)
        #trg_repeat is now [batch_size, beam_size, seq_len, 1]
        word_probs = beam_out.gather(3, trg_repeat).squeeze(-1)
        mask = (~trg_repeat.eq(1)).float().squeeze(-1)
        reward = torch.ones(reward.shape).float().to(DEVICE)
   
    #meritocratic update
    q = compute_q(beam_out, unmasked_beam_words, beta, reward)

    word_probs = mask*word_probs
    word_probs = word_probs.sum(-1)#[batch, beam]
    loss = word_probs*q*reward
    loss = -loss.mean()
    return loss 

def forward_pass(batch, model, epoch_num=0,
                 src_field=None, trg_field=None,
                 require_eos=True, ignore_gen=None, args=None, max_len=None,
                 length_cache=None, score_cache=None):
    '''
        return loss
    '''
    loss = 0
    train_mode=args["train_mode"]
    beta=args["beta"]
    score_type=args["score_type"]
    loss_type=args["loss_type"]
    baseline=args["baseline"]
    teacher_force=args["teacher_force"]
    planner_path=args["planner_path"]
    alpha = 1.0
    if "alpha" in args:
        alpha = args["alpha"]

    nl_trg, nl_trg_lengths = None, None

    mr_out, mr_words, nl_out, nl_words = None, None, None, None
    ignore_gen=loss_type=="parse"#NOTE
    start = time.time()
    model_out = model.forward(batch["nl_src"], 
                              batch["acts_trg"],
                              batch["nl_src_lengths"], 
                              batch["acts_trg_lengths"],
                              batch["ltl_data"],
                              teacher_force=teacher_force,
                              nl_trg=nl_trg,
                              nl_trg_lengths=nl_trg_lengths,
                              trg_field=trg_field,
                              src_field=src_field,
                              require_eos=require_eos,
                              ignore_gen=ignore_gen,
                              args=args, max_len=max_len)

    out, actions = model_out
    criterion = nn.NLLLoss(reduction="mean")
    loss = criterion(out.view(-1, 5), batch["acts_trg"].view(-1))
    return loss

def run_epoch(data_iter, model, optim,
              epoch_num=0, require_eos=True,
              src_field=None, trg_field=None,
              ignore_gen=False, args=None, max_len=None,
              length_cache=None, score_cache=None):
    start = time.time()
    total_loss = 0
    print_tokens = 0
    for i, batch in enumerate(data_iter, 1):
        start = time.time()
        loss = forward_pass(batch, model,
                            epoch_num=epoch_num,
                            src_field=src_field, trg_field=trg_field,
                            ignore_gen=ignore_gen,
                            args=args, max_len=max_len,
                            length_cache=length_cache, score_cache=score_cache)
        #print_memory_usage("forward_pass", device=batch["nl_trg"].device)
        #print()
        start = time.time()
        old_weights = {}
        loss.backward()
        optim.step()
        optim.zero_grad()
        #print_timing("loss.backward", (time.time()-start), 0)
           
        total_loss += loss.detach().item()

        print_tokens += batch["ntokens"]
    return total_loss

def train(model, optim, train_iter, valid_iter,
          src_field=None, trg_field=None, 
          verbose=True, log=True, ignore_gen=False, args=None,
          ltl_args=True, max_len=None, length_cache=None, score_cache=None,
          curriculum=None):

    assert not args["train_mode"]=="rl" or args["decode_mode"] is not None
    test_flag = True
    num_epochs = args["num_epochs"]
    checkpoint = args["checkpoint"]
    score_type=args["score_type"]
    max_score = 0
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        start = time.time()
        model.train()

        train_max_len = None
        c_args = args.copy()
        if curriculum:
            for c in curriculum:
                if c["min_epoch"] <= epoch < c["max_epoch"]:
                    for c_arg in c:
                        if c_arg == "max_len":
                            train_max_len = c["max_len"]
                        else:
                            c_args[c_arg] = c[c_arg]

        loss = run_epoch(train_iter, 
                         model, optim,
                         epoch_num=epoch,
                         src_field=src_field,
                         trg_field=trg_field,
                         require_eos=False,
                         ignore_gen=ignore_gen,
                         args=c_args, max_len=train_max_len,
                         length_cache=length_cache, score_cache=score_cache)
        writer.add_scalar(f'train_loss', loss, epoch)
        print_timing("run_epoch time", (time.time() - start), 0)
        print(f'train_loss', loss)

        model.eval()
        with torch.no_grad():
            if log:
                start = time.time()
                metric_args = c_args.copy()
                metric_args["beam_width"] = 10
                metrics = get_action_metrics(valid_iter, model,
                                             trg_field, src_field,
                                             device=DEVICE, args=metric_args,
                                             ltl_args=ltl_args, max_len=train_max_len)
                acceptance = metrics
                writer.add_scalar(f'acceptance', acceptance, epoch)
                print(f'acceptance', acceptance)
                print_timing("compute_accuracy time", (time.time() - start), 0)
            if verbose:
                start = time.time()
                print_args = c_args.copy()
                print_args["beam_width"] = 10
                print_args["valid_only"] = True#NOTE
            if (epoch%10 == 0 or epoch == num_epochs-1) and checkpoint:
                torch.save(model.state_dict(), f'{OUT_DIR}/checkpoint_{epoch}.pth')
            if log and acceptance > max_score and checkpoint:
                max_score = acceptance
                torch.save(model.state_dict(), f'{OUT_DIR}/best_checkpoint.pth')
            if length_cache is not None and score_cache is not None:
                pickle.dump(length_cache, open(f'{OUT_DIR}/length_cache.pkl', 'wb'))
                pickle.dump(score_cache, open(f'{OUT_DIR}/{score_type}_cache.pkl', 'wb'))

def get_action_trg(ltl_json):
    data = ltl_json["data"]
    lines = []
    for d in data:
        seq = d["actions"][0]["steps"][0]
        seq = " ".join(map(str, seq))
        lines.append(seq)
    return lines

if __name__=="__main__":
    with open(cfg["data_dir"] + "train.src") as f:
        src_lines = f.readlines()
    with open(cfg["data_dir"] + "train_ltl_data.json") as f:
        ltl_json = json.load(f)
        trg_lines = get_action_trg(ltl_json)

    with open(cfg["data_dir"] + "test.src") as f:
        test_src_lines = f.readlines()
    with open(cfg["data_dir"] + "test_ltl_data.json") as f:
        test_ltl_json = json.load(f)
        test_trg_lines = get_action_trg(test_ltl_json)

    if cfg["sort"]:
        src_trg = list(zip(src_lines, trg_lines, ltl_json["data"]))
        src_trg = sorted(src_trg, key=lambda x: len(x[0]))
        src_lines, trg_lines, ltl_json["data"] = zip(*src_trg)

    src_field = Field("src", unk_token="<unk>", pad_token="<pad>", 
                            sos_token="<s>", eos_token="</s>")
    trg_field = Field("trg")

    src_field.build_vocab(src_lines)
    trg_field.build_vocab(["0","1","2","3","4"])
    my_data = {}

    split_sizes = [cfg["train_split"], cfg["val_split"]]
    split_range = zip([sum(split_sizes[:i]) for i in range(len(split_sizes))],
                      [sum(split_sizes[:i+1]) for i in range(len(split_sizes))])
    splits = ["train", "val"]
    train_src_lines = []

    for split, part in zip(splits, split_range):
        split_ltl = {"data": ltl_json["data"][part[0]:part[1]]}
        ids = list(range(part[0], part[1]))
        my_data[split] = ActionDataset(src_lines[part[0]:part[1]], 
                                      trg_lines[part[0]:part[1]],
                                      src_field, trg_field,
                                      ltl_json=split_ltl,
                                      ids=ids)
        if split == "train":
           train_src_lines = src_lines[part[0]:part[1]] 
        with open(f'{OUT_DIR}/{split}.src', 'w') as f:
            f.write('\n'.join(src_lines[part[0]:part[1]]))
        with open(f'{OUT_DIR}/{split}.trg', 'w') as f:
            f.write('\n'.join(trg_lines[part[0]:part[1]]))

    train_val_n = len(src_lines)
    ids = list(range(train_val_n, train_val_n + len(test_src_lines)))
    my_data["test"] = ActionDataset(test_src_lines, 
                                   test_trg_lines,
                                   src_field, trg_field,
                                   ltl_json=test_ltl_json,
                                   ltl=True,
                                   ids=ids)

    my_collator = ActionCollator(pad_idx=1, device=DEVICE)


    my_loaders = {}
    for split in ["train", "val", "test"]:
        shuffle=False
        if split in ["train"]:
            batch_size = cfg["train_batch_size"] if cfg["train_mode"]!="ml" else cfg["ml_train_batch_size"]
            #shuffle = True
        else:
            batch_size = cfg["val_batch_size"]
        my_loaders[split] = DataLoader(my_data[split], 
                                       batch_size=batch_size,
                                       collate_fn=my_collator,
                                       shuffle=shuffle)
    #initialize model
    model = make_seq2seq_base(emb_size=200, 
                              hidden_size=cfg['hidden_size'],
                              num_layers=2, dropout=cfg['dropout'],
                              src_field=src_field, trg_field=trg_field)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) 
    model.to(DEVICE)

    pad_idx = src_field.vocab.pad_idx
    criterion = nn.NLLLoss(reduction="mean", ignore_index=pad_idx)
    adam_optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    sgd_optim = torch.optim.SGD(model.parameters(), lr=cfg['lr'])
    rms_optim = torch.optim.RMSprop(model.parameters(), lr=cfg['lr'])
    args = cfg
    ltl_args = get_args()

    max_len = 20 if cfg["train_mode"] != "ml" else None#TODO
    score_cache, explore_cache = None, None
    if cfg["load_pretrained"]:
        model.load_state_dict(torch.load(cfg["pretrained"]))

    length_cache, score_cache, explore_cache = {}, {}, {}

    if cfg["eval_only"]:
        assert cfg["load_pretrained"]
        metric_args = args.copy()
        print(metric_args["pretrained"])
        for split in ["val", "test", "train"]:
        #for split in ["test"]:
            metrics = get_action_metrics(my_loaders[split], model,
                                         trg_field, src_field,
                                         device=DEVICE, args=metric_args,
                                         ltl_args=ltl_args, max_len=max_len,
                                         get_execute=True)
            acceptance = metrics
            print(f'{split} acceptance', acceptance, str(round(acceptance, 3)))
            print()
        exit()

    print_data_info(my_data, src_field, trg_field)

    train(model, adam_optim, 
          my_loaders["train"], my_loaders["val"], 
          src_field=src_field, 
          trg_field=trg_field,
          args=args,
          ltl_args=ltl_args,
          max_len=max_len,
          length_cache=length_cache, 
          score_cache=score_cache,
          curriculum=None)

    torch.save(model.state_dict(), f'{OUT_DIR}/trained_model.pth')
    eval_args = args.copy()
    eval_args["decode_mode"]="beam"
    eval_args["beam_width"]=10
    writer.close()
