#!/usr/bin/env python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from dataset import TextCollator, ParseDataset
from utils import *
from rewards import *
from model import make_parser, make_autoencoder

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

def explore_batch(batch, model, epoch_num=0,
                  src_field=None, trg_field=None, cache=None,
                  reward_type=None, args=None, length_cache=None, 
                  score_cache=None, max_len=None):
    auto_enc=args["model_type"]=="auto_enc"
    planner_path=args["planner_path"]
    score_type=args["score_type"]
    curr_model = model
    mr_out, mr_words, nl_out, nl_words = None, None, None, None

    #TODO
    #mr_words = batch["mr_trg_y"].unsqueeze(1).expand(-1, 3, -1)
    #mr_lengths = batch["mr_trg_lengths"].unsqueeze(1).expand(-1, 3)
    #rewards = compute_ltl_rewards(mr_words, mr_lengths,
    #                              batch["ltl_data"],
    #                              trg_field, 
    #                              ltl_args,
    #                              device=DEVICE,
    #                              trg=batch["mr_trg_y"],
    #                              ids=batch["id"],
    #                              length_cache=length_cache,
    #                              score_cache=score_cache,
    #                              score_type=score_type,
    #                              model_path=planner_path )
    #exit()
    #/TODO

    if auto_enc:
        curr_model = model.parser
    if "parse" in reward_type:
        model_out = curr_model.forward(batch["nl_src"], 
                                       batch["mr_trg"],
                                       batch["nl_src_lengths"], 
                                       batch["mr_trg_lengths"],
                                       nl_trg=batch["nl_trg"],
                                       nl_trg_lengths=batch["nl_trg_lengths"],
                                       trg_field=trg_field,
                                       src_field=src_field,
                                       ignore_gen=reward_type=="parse",#NOTE
                                       args=args, max_len=max_len,
                                       teacher_force=False)
        mr_out, mr_words, attns = model_out
    #print(mr_words.shape)
    #print([trg_field.lookup_words(x) for x in mr_words[0]])
    if "gen" in reward_type:
        #print(mr_words.shape)
        gen_args = args.copy()
        gen_args["teacher_force"] = True
        model_out = gen_forward_pass(mr_out, mr_words, None, model,
                                     nl_field=src_field,
                                     mr_field=trg_field,
                                     require_eos=False, 
                                     args=gen_args, max_len=None,
                                     nl_trg=batch["nl_trg"],
                                     nl_trg_lengths=batch["nl_trg_lengths"])

        nl_out, nl_words, _ = model_out

    #attns is [batch, beam, pred_seq_len, src_seq_len]
    assert not (mr_out != mr_out).any(), 'NaN/inf in mr probs'

    mr_words, mr_lengths = batch_output(mr_words, trg_field, device=DEVICE)
    #mr_words is [batch_size, beam_size, seq_len]

    baseline=args["baseline"]

    #print(reward_type)
    rewards = None
    if "parse" in reward_type:
        rewards = compute_ltl_rewards(mr_words, mr_lengths,
                                      batch["ltl_data"],
                                      trg_field, 
                                      ltl_args,
                                      device=DEVICE,
                                      trg=batch["mr_trg_y"],
                                      ids=batch["id"],
                                      length_cache=length_cache,
                                      score_cache=score_cache,
                                      score_type=score_type,
                                      model_path=planner_path )

    if "gen" in reward_type:
        #print(src_field.lookup_words(batch["nl_trg_y"][0]))
        #print()
        #if nl_words is not None:
        #    for words in nl_words[0]:
        #        print(src_field.lookup_words(words))
        nl_rewards = compute_nl_rewards(mc_nl_out=nl_out, mc_nl_words=nl_words, 
                                        nl_trg=batch["nl_trg_y"],
                                        nl_lengths=batch["nl_trg_lengths"],
                                        nl_field=src_field)
        rewards = rewards + nl_rewards
    rewards = rewards.detach()
    max_reward, arg_max = rewards.max(-1, keepdim=True)
    #print(max_reward)
    batch_size, _, seq_len = mr_words.shape
    #mr_words [batch, beam, seq]
    #arg_max [batch, 1]
    best_seqs = mr_words[torch.arange(batch_size).unsqueeze(-1), arg_max]
    best_seqs = best_seqs.detach().squeeze(1)
    best_seqs = best_seqs.contiguous()
    src_lines, trg_lines, new_src_lines = [], [], []
    ex_ids = batch["id"]
    best_ltl_data = []
    for i in range(batch["nl_src"].shape[0]):
        src_lines.append(src_field.get_string(batch["nl_src"][i]))
    for i in range(best_seqs.shape[0]):
        ex_id = ex_ids[i] 
        idxs = best_seqs[i] 
        score = max_reward[i][0].item()
        baseline_score = 10**(-15)
        if score > cache.get(ex_id, (baseline_score,))[0]:
            max_trg = trg_field.get_string(idxs.tolist())
            #print("better than cache!")
            #if ex_id in cache:
            #    print("old", ex_id)
            #    print(cache[ex_id][:3])
            new_ltl_data = batch["ltl_data"][i]
            cache[ex_id] = (score, src_lines[i], max_trg, new_ltl_data)
            #print(cache[ex_id][:3])
            #print(postfix2infix(max_trg, trg_field, typed=True))
    for i in range(best_seqs.shape[0]):
        ex_id = ex_ids[i] 
        if cache.get(ex_id, 0):
            #if cache[ex_id][2] == "workbench G !":
            #    print(cache[ex_id][2])
            #    print(cache[ex_id][1])
            #    exit()
            #print(cache[ex_id][0])
            #print(cache[ex_id][1])
            #print(cache[ex_id][2])
            #print()
            trg_lines.append(cache[ex_id][2])
            best_ltl_data.append(cache[ex_id][3])
            new_src_lines.append(cache[ex_id][1])
    return new_src_lines, trg_lines, best_ltl_data

def explore(data_iter, model, epoch_num=0, src_field=None, trg_field=None,
            reward_type="parse", cache=None, args=None, length_cache=None,
            max_len=None, score_cache=None):
    '''
        model is EncoderDecoder
    '''
    trg_lines = []
    new_src_lines = []
    best_ltl_data = {"data":[]}
    for i, batch in enumerate(data_iter, 1):
        new_src, trg, ltl = explore_batch(batch, model, reward_type=reward_type,
                                     cache=cache, 
                                     src_field=src_field, trg_field=trg_field,
                                     args=args, length_cache=length_cache,
                                     score_cache=score_cache,
                                     max_len=max_len)
        new_src_lines.extend(new_src)
        trg_lines.extend(trg)
        best_ltl_data["data"].extend(ltl)
    #print(new_src)
    #print("here one")
    #for s, t in zip(new_src_lines, trg_lines):
    #    print(s)
    #    print(t)
    #    print()
    #exit()
    return new_src_lines, trg_lines, best_ltl_data

def forward_pass(batch, model, epoch_num=0,
                 src_field=None, trg_field=None,
                 require_eos=True, ignore_gen=None, args=None, max_len=None,
                 length_cache=None, score_cache=None):
    '''
        return loss
    '''
    loss = 0
    auto_enc=args["model_type"]=="auto_enc"
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
    auto_enc = args["model_type"]=="auto_enc"
    if auto_enc:
        nl_trg, nl_trg_lengths = batch["nl_trg"], batch["nl_trg_lengths"]

    mr_out, mr_words, nl_out, nl_words = None, None, None, None
    curr_model = model
    #if auto_enc:#TODO
    #    curr_model = model.parser
    ignore_gen=loss_type=="parse"#NOTE
    #print(ignore_gen)
    start = time.time()
    model_out = curr_model.forward(batch["nl_src"], 
                              batch["mr_trg"],
                              batch["nl_src_lengths"], 
                              batch["mr_trg_lengths"],
                              teacher_force=teacher_force,
                              nl_trg=nl_trg,
                              nl_trg_lengths=nl_trg_lengths,
                              trg_field=trg_field,
                              src_field=src_field,
                              require_eos=require_eos,
                              ignore_gen=ignore_gen,
                              args=args, max_len=max_len)

    #print(auto_enc)
    #print_timing("forward_pass", (time.time()-start), 1)
    if auto_enc and not ignore_gen:
        mr_out, mr_words, _, nl_out, nl_words, _ = model_out
    else:
        mr_out, mr_words, _ = model_out
#    print(mr_words.shape)
    #print_memory_usage("parser forward", device=batch["nl_trg"].device)
    
    #print(mr_out.shape)
    #print(nl_out.shape)
    #print_memory_usage("gen_parser forward", device=batch["nl_trg"].device)
    mr_loss = 0
    if "parse" in loss_type:
        unmasked_mr_words = mr_words.clone()
        mr_words, mr_lengths = batch_output(mr_words, trg_field, device=DEVICE)
        #mr_words is [batch_size, beam_size, seq_len]

        start = time.time()
        rewards = torch.ones(mr_words.shape[:-1])
        if train_mode != "ml":
            rewards = compute_ltl_rewards(mr_words, mr_lengths,
                                          batch["ltl_data"],
                                          trg_field, 
                                          ltl_args,
                                          device=DEVICE,
                                          trg=batch["mr_trg_y"],
                                          ids=batch["id"],
                                          length_cache=length_cache,
                                          score_cache=score_cache,
                                          score_type=args["score_type"], 
                                          model_path=planner_path)
            #[batch_size, beam_size]
            #print(rewards.sum())
        #print_timing("compute_rewards", (time.time()-start), 1)

        mr_loss = compute_loss(mr_out, mr_words, mr_lengths, rewards, 
                               batch["mr_trg_y"], train_mode=train_mode,
                               unmasked_beam_words=unmasked_mr_words, beta=beta)
    nl_loss = 0
    #assert loss_type != "parse_gen"
    if "gen" in loss_type:
        start = time.time()
        nl_loss = compute_nl_loss(mc_nl_out=nl_out, mc_nl_words=nl_words, 
                                  nl_trg=batch["nl_trg_y"],
                                  nl_lengths=batch["nl_trg_lengths"])
        #print_timing("compute_nl_loss", (time.time()-start), 1)
    return alpha*mr_loss + nl_loss

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

def eval_val(file_name, model, data_iter, src_field, trg_field, 
             datasets, device=None, args=None, max_len=None):
    '''
        Run evaluation for the examples in data_iter and
        write the predictions to file_name
    '''
    auto_enc=args["model_type"]=="auto_enc"
    model.eval()
    n= float('inf') #get all examples
    src_eos_index = src_field.vocab.stoi[src_field.vocab.eos_token]
    trg_eos_index = trg_field.vocab.stoi[trg_field.vocab.eos_token]
    sos_index = src_field.vocab.stoi[trg_field.vocab.sos_token]
    string_out = get_string_rep(data_iter, model,
                                n=n, sos_index=sos_index,
                                src_eos_index=src_eos_index,
                                trg_eos_index=trg_eos_index,
                                src_field=src_field,
                                trg_field=trg_field,
                                device=device,
                                args=args,
                                max_len=max_len)       
    if auto_enc:
        src_nls, trg_mrs, pred_mrs, pred_nls = string_out
    else:
        src_nls, trg_mrs, pred_mrs = string_out

    string_reps = zip(src_nls, trg_mrs, pred_mrs)
    with open(file_name, "w") as file:
        for batch in string_reps:
            file.write("src:  "+batch[0]+"\n")
            file.write("trg:  "+batch[1]+"\n")
            file.write("pred: "+batch[2]+"\n")
            file.write("\n")

def iter_ml(model, optim, train_iter, valid_iter,
            src_field=None, trg_field=None,  
            train_src_lines=None,
            my_collator=None, num_gpus=0, verbose=True, log=True,
            args=None, max_len=None, explore_cache=None, score_cache=None,
            length_cache=None, curriculum=None):
    """Train a model on iterative ml"""    
    beam_width=args["beam_width"]
    epsilon=args["epsilon"]
    valid_only=args["valid_only"]
    baseline=args["baseline"]
    beta=args["beta"]
    decode_mode=args["decode_mode"]
    batch_size=args["train_batch_size"]
    num_epochs=args["num_epochs"] 
    score_type=args["score_type"]
    auto_enc=args["model_type"]=="auto_enc"
    loss_type=args["loss_type"]
        
    cache = explore_cache

    max_score = 0
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        reward_type = "parse_gen"#TODO
        model.eval()

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


        epoch_start = time.time()
        with torch.no_grad():
            start = time.time()
            explore_args = c_args.copy()
            explore_args["teacher_force"] = False
            src_lines, trg_lines, new_ltl = explore(train_iter, 
                                                    model,
                                                    epoch_num=epoch,
                                                    src_field=src_field,
                                                    trg_field=trg_field,
                                                    reward_type=reward_type,
                                                    cache=cache,
                                                    length_cache=length_cache,
                                                    args=explore_args,
                                                    score_cache=score_cache,
                                                    max_len=train_max_len)
            #print_timing("explore", time.time() - start, 0)
        model.train()
        start = time.time()
        #print(len(src_lines), num_gpus)
        if len(src_lines) > num_gpus:
            #for t in trg_lines:
            #    print("testing", f'\"{t}\"')
            #    print(postfix2infix(t, trg_field)[0])
            trg_lines = [postfix2infix(t, trg_field)[0] for t in trg_lines]
            best_dataset = ParseDataset(src_lines, 
                                        trg_lines,
                                        src_field, trg_field,
                                        ltl_json=new_ltl,
                                        ltl=True)

            best_dataloader = DataLoader(best_dataset, 
                                         batch_size=c_args["ml_train_batch_size"],
                                         collate_fn=my_collator,
                                         shuffle=False)
            ml_args = c_args.copy()
            ml_args["num_epochs"] = 10
            ml_args["beam_width"] = 1
            ml_args["train_mode"] = "ml"
            ml_args["epsilon"] = 0
            ml_args["checkpoint"] = False
            ml_args["teacher_force"] = False#NOTE change if "ml"
            ml_args["decode_mode"] = "greedy"#NOTE could also be "ml"
            #ml_args["loss_type"] = "parse" #NOTE important for ignore_gen
            adam_optim = torch.optim.Adam(model.parameters(), lr=args['lr'])
            train(model, adam_optim, 
                  best_dataloader, valid_iter,
                  src_field=src_field, trg_field=trg_field,
                  verbose=False, log=False,
                  args=ml_args,
                  max_len=None)#NOTE for ML
        model.eval()
        with torch.no_grad():
            checkpoint = c_args["checkpoint"]
            if log:
                metric_args = c_args.copy()
                metric_args["beam_width"] = 10
                metrics = get_ltl_metrics(valid_iter, model,
                                          trg_field, src_field,
                                          device=DEVICE, args=metric_args,
                                          ltl_args=ltl_args, max_len=train_max_len)
                f1, fsa_score, acceptance, execute = metrics
                writer.add_scalar(f'string_f1', f1, epoch)
                writer.add_scalar(f'fsa_score', fsa_score, epoch)
                writer.add_scalar(f'acceptance', acceptance, epoch)
                print_timing("compute_accuracy time", (time.time() - start), 0)
                print(f'string_f1', f1)
                print(f'fsa_score', fsa_score)
                print(f'acceptance', acceptance)
            if verbose:
                print("val examples")
                print_args = c_args.copy()
                print_args["beam_width"] = 10
                print_args["valid_only"] = True
                print_examples(valid_iter, model, n=10,
                               src_field=src_field, trg_field=trg_field,
                               device=DEVICE,
                               file_name=f'{OUT_DIR}/iml_epoch_{epoch}.txt',
                               args=print_args, max_len=train_max_len)
            if (epoch%10 == 0 or epoch == num_epochs-1) and checkpoint:
                torch.save(model.state_dict(), 
                           f'{OUT_DIR}/iml_checkpoint_{epoch}.pth')
            if log and acceptance > max_score and checkpoint:
                max_score = acceptance
                torch.save(model.state_dict(), f'{OUT_DIR}/iml_best_checkpoint.pth')
            pickle.dump(cache, open(f'{OUT_DIR}/explore_cache.pkl', 'wb'))
            pickle.dump(length_cache, open(f'{OUT_DIR}/length_cache.pkl', 'wb'))
            pickle.dump(score_cache, open(f'{OUT_DIR}/{score_type}_cache.pkl', 'wb'))
        print((time.time() - epoch_start)/60)

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
                metrics = get_ltl_metrics(valid_iter, model,
                                          trg_field, src_field,
                                          device=DEVICE, args=metric_args,
                                          ltl_args=ltl_args, max_len=train_max_len)
                f1, fsa_score, acceptance, execute = metrics
                writer.add_scalar(f'string_f1', f1, epoch)
                writer.add_scalar(f'fsa_score', fsa_score, epoch)
                writer.add_scalar(f'acceptance', acceptance, epoch)
                print_timing("compute_accuracy time", (time.time() - start), 0)
                print(f'string_f1', f1)
                print(f'fsa_score', fsa_score)
                print(f'acceptance', acceptance)
            if verbose:
                print("val examples")
                start = time.time()
                print_args = c_args.copy()
                print_args["beam_width"] = 10
                print_args["valid_only"] = True#NOTE
                print_examples(valid_iter, model, n=10,
                               src_field=src_field, trg_field=trg_field,
                               require_eos=True,#NOTE 
                               device=DEVICE,
                               file_name=f'{OUT_DIR}/ml_current_out.txt',
                               args=print_args, max_len=train_max_len)
                print_timing("print_examples time", (time.time() - start), 0)
            if (epoch%10 == 0 or epoch == num_epochs-1) and checkpoint:
                torch.save(model.state_dict(), f'{OUT_DIR}/checkpoint_{epoch}.pth')
            if log and acceptance > max_score and checkpoint:
                max_score = acceptance
                torch.save(model.state_dict(), f'{OUT_DIR}/best_checkpoint.pth')
            if length_cache is not None and score_cache is not None:
                pickle.dump(length_cache, open(f'{OUT_DIR}/length_cache.pkl', 'wb'))
                pickle.dump(score_cache, open(f'{OUT_DIR}/{score_type}_cache.pkl', 'wb'))

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
        ids = list(range(part[0], part[1]))
        my_data[split] = ParseDataset(src_lines[part[0]:part[1]], 
                                      trg_lines[part[0]:part[1]],
                                      src_field, trg_field,
                                      ltl_json=split_ltl,
                                      ltl=True,
                                      ids=ids)
        if split == "train":
           train_src_lines = src_lines[part[0]:part[1]] 
        with open(f'{OUT_DIR}/{split}.src', 'w') as f:
            f.write('\n'.join(src_lines[part[0]:part[1]]))
        with open(f'{OUT_DIR}/{split}.trg', 'w') as f:
            f.write('\n'.join(trg_lines[part[0]:part[1]]))

    train_val_n = len(src_lines)
    ids = list(range(train_val_n, train_val_n + len(test_src_lines)))
    my_data["test"] = ParseDataset(test_src_lines, 
                                   test_trg_lines,
                                   src_field, trg_field,
                                   ltl_json=test_ltl_json,
                                   ltl=True,
                                   ids=ids)

    my_collator = TextCollator(pad_idx=1, device=DEVICE)


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
    if cfg["model_type"]=="auto_enc":
        model = make_autoencoder(emb_size=200, hidden_size=cfg['hidden_size'],
                                 num_layers=2, dropout=cfg['dropout'],
                                 nl_field=src_field, mr_field=trg_field)
    else:
        model = make_parser(emb_size=200, hidden_size=cfg['hidden_size'],
                             num_layers=2, dropout=cfg['dropout'],
                             src_field=src_field, trg_field=trg_field)

    num_gpus = torch.cuda.device_count()
    if cfg["model_type"]=="auto_enc":
        if num_gpus > 1:
            model.generator = nn.DataParallel(model.generator)
            model.parser = nn.DataParallel(model.parser)
        model.generator.to(DEVICE)
        model.parser.to(DEVICE)
    else:
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

    max_len = 15 if cfg["train_mode"] != "ml" else None#TODO
    score_cache, explore_cache = None, None
    if cfg["load_pretrained"]:
        model.load_state_dict(torch.load(cfg["pretrained"]))

    length_cache, score_cache, explore_cache = {}, {}, {}
    pickle_path = cfg["pickle_path"]
    score_type = cfg["score_type"]
    if cfg["load_pretrained_score_cache"]:
        score_cache = pickle.load(open(pickle_path+f'{score_type}_cache.pkl', "rb"))
    if cfg["load_pretrained_explore_cache"]:
        explore_cache = pickle.load(open(pickle_path+"explore_cache.pkl", "rb"))
    if cfg["load_pretrained_length_cache"]:
        length_cache = pickle.load(open(pickle_path+"length_cache.pkl", "rb"))

    if cfg["eval_only"]:
        assert cfg["load_pretrained"]
        metric_args = args.copy()
        metric_args["beam_width"] = 10
        print(metric_args["pretrained"])
        for split in ["val", "test", "train"]:
        #for split in ["test"]:
            model.eval()
            print_args = args.copy()
            print_args["beam_width"] = 10
            print_args["valid_only"] = True
            print_examples(my_loaders[split], model, n=10,
                           src_field=src_field, trg_field=trg_field,
                           device=DEVICE,
                           file_name=f'{OUT_DIR}/{split}.txt',
                           args=print_args, max_len=15)#NOTE hardcode
            metrics = get_ltl_metrics(my_loaders[split], model,
                                      trg_field, src_field,
                                      device=DEVICE, args=metric_args,
                                      ltl_args=ltl_args, max_len=max_len,
                                      get_execute=True)
            f1, fsa_score, acceptance, execute = metrics
            print(f'{split} f1', f1, str(round(f1, 3)))
            print(f'{split} fsa_score', fsa_score, str(round(fsa_score, 3)))
            print(f'{split} acceptance', acceptance, str(round(acceptance, 3)))
            print(f'{split} execution', execute, str(round(execute, 3)))

            reward_metrics = ltl_reward_metrics(my_loaders[split], model,
                                                trg_field, src_field,
                                                length_cache, score_cache, ids,
                                                device=DEVICE, args=metric_args,
                                                ltl_args=ltl_args, max_len=max_len)
            #f1, fsa_score, acceptance, execute = metrics
            pred_reward, gt_reward = reward_metrics
            print(f'{split} avg pred reward', pred_reward, str(round(pred_reward, 3)))
            print(f'{split} avg gt reward', gt_reward, str(round(gt_reward, 3)))
            print()
        exit()

    if "repl" in cfg and cfg["repl"]:
        while True:
            nl = input("sentence: ")
            singleton_dataset = ParseDataset([nl], 
                                             ["pear"],#NOTE doesn't matter
                                             src_field, trg_field,#NOTE doesn't matter
                                             ltl_json=test_ltl_json,#NOTE doesn't matter
                                             ltl=True)

            singleton_loader = DataLoader(singleton_dataset, 
                                          batch_size=1,
                                          collate_fn=my_collator,
                                          shuffle=shuffle)
            batch = next(iter(singleton_loader))
            repl_args = args.copy()
            repl_args["beam_width"] = 10
            repl_args["decode_mode"] = "beam"
            repl_args["epsilon"] = 0
            repl_args["valid_only"] = True
            model_out = model.forward(batch["nl_src"], 
                                      batch["mr_trg"],
                                      batch["nl_src_lengths"], 
                                      batch["mr_trg_lengths"],
                                      teacher_force=False,
                                      nl_trg=batch["nl_trg"],
                                      nl_trg_lengths=batch["nl_trg_lengths"],
                                      trg_field=trg_field,
                                      src_field=src_field,
                                      require_eos=True,
                                      ignore_gen=True,
                                      args=repl_args, max_len=17)
            #print(auto_enc)
            #print_timing("forward_pass", (time.time()-start), 1)
            mr_out, mr_words, _ = model_out
            mr = trg_field.get_string(mr_words[0][0], 
                                 strip_pad=True)
            mr = postfix2infix(mr, trg_field)
            print(mr[0])

    print_data_info(my_data, src_field, trg_field)

    curriculum_path = args["curriculum_path"]
    curriculum = json.load(open(curriculum_path, "r"))
    num_epochs = max([c["max_epoch"] for c in curriculum])
    args["num_epochs"] = num_epochs

    if cfg["train_mode"] == "iml":
        iter_ml(model, adam_optim, 
                my_loaders["train"], my_loaders["val"], 
                src_field=src_field, 
                trg_field=trg_field,
                train_src_lines=train_src_lines,
                my_collator=my_collator,
                num_gpus=num_gpus,
                args=args, max_len=max_len, 
                explore_cache=explore_cache, score_cache=score_cache,
                length_cache=length_cache,
                curriculum=curriculum)
    elif cfg["train_mode"] in ["rl", "ml"]:
        train(model, adam_optim, 
              my_loaders["train"], my_loaders["val"], 
              src_field=src_field, 
              trg_field=trg_field,
              args=args,
              ltl_args=ltl_args,
              max_len=max_len,
              length_cache=length_cache, 
              score_cache=score_cache,
              curriculum=curriculum)
    else:
        print("unknown train mode")

    torch.save(model.state_dict(), f'{OUT_DIR}/trained_model.pth')
    eval_args = args.copy()
    eval_args["decode_mode"]="beam"
    eval_args["beam_width"]=10
    eval_val(f'{OUT_DIR}/train.pred', model, my_loaders["train"], 
             src_field, trg_field, my_data, device=DEVICE,
             args=eval_args, max_len=15)
    eval_val(f'{OUT_DIR}/val.pred', model, my_loaders["val"], 
             src_field, trg_field, my_data, device=DEVICE,
             args=eval_args, max_len=15)

    writer.close()
