#!/usr/bin/env python
import subprocess
import random
import re
import sacrebleu
import torch
import math
import operator

from functools import reduce
from rouge import Rouge
from utils import *
from ltl.ltl2tree import replace_symbols, ltl2tree
from ltl.algo.a2c_acktr import A2C_ACKTRTrainer
from ltl.worlds.craft_world import Cookbook


def get_syntax_scores(x, trg_field):
    '''
        input
            x [beam_size, batch_size, pred_seq_len]
        out
            [beam_size*batch_size]
    '''
    beam_size, batch_size, pred_seq_len = x.shape
    x_flat = x.contiguous().view(-1, pred_seq_len)

    x_strings = []
    for i in range(beam_size*batch_size):
        x_strings.append(trg_field.get_string(x_flat[i]).strip())

    scores = []
    for x_string in x_strings:
        prev_digit = False
        x_words = x_string.split(" ")
        score = 1
        if len(x_words) <= 1:
            score = 0
        if x_words[0].isdigit():
            score = 0
        for i in range(len(x_words)):
            if i > 1 and x_words[i-2].isdigit()\
                     and x_words[i-1].isdigit()\
                     and x_words[i].isdigit():
                score = 0
            if i > 0 and not x_words[i-1].isdigit() and not x_words[i].isdigit():
                score = 0
            if x_words[i]=="<pad>" or x_words[i]=="<unk>":
                score = 0
        scores.append(score)
    return scores 


def get_bleu_scores(trg, x, trg_field):
    '''
        input
            trg [beam_size, batch_size, trg_seq_len]
            x [beam_size, batch_size, pred_seq_len]
        out
            [beam_size*batch_size]
    '''
    beam_size, batch_size, seq_len = trg.shape
    x_flat = x.contiguous().view(-1, seq_len)
    y_flat = trg.contiguous().view(-1, seq_len)


    x_strings, y_strings = [], []
    for i in range(beam_size*batch_size):
        x_string = trg_field.get_string(x_flat[i])
        x_strings.append(x_string)
        y_strings.append(trg_field.get_string(y_flat[i]))

    scores = []
    rouge = Rouge()
    for sys, ref in zip(x_strings, y_strings):
        b = sacrebleu.corpus_bleu([sys], [[ref]]).score
        #r2 = rouge.get_scores(sys + " ", ref)[0]["rouge-2"]["r"]
        #r1 = rouge.get_scores(sys + " ", ref)[0]["rouge-1"]["r"]
        score = b/100
        scores.append(score)
    return scores 

def object_recog_score(repeated_trg, beam_major, trg_mask, trg_lengths, trg_field):
    '''
    input
        repeated_trg = [beam, batch_size, seq_len]
        beam_major = [beam_size, batch_size, seq_len]
        trg_mask = [beam_size, batch_size, seq_len]
        trg_lengths = [beam, batch]
    returns
        [beam, batch] tensor of rewards
    '''
    objects = ["person", "chair", "hallway", "bag", "table", "apple", "book", 
               "floor", "car", "telescope", "cars", "pants"]
    object_idx = set([trg_field.vocab.stoi[o] for o in objects])
    trg_flat = repeated_trg.contiguous().view(-1, repeated_trg.shape[-1])
    pred_flat = beam_major.contiguous().view(-1, beam_major.shape[-1])
    scores = []
    for i in range(trg_flat.shape[0]):
        trg = trg_flat[i]
        pred = pred_flat[i]
        pred_obj = set()
        trg_obj = set()
        for j in range(trg_flat.shape[1]):
            if trg[j].item() in object_idx:
                trg_obj.add(trg[j].item())
            if pred[j].item() in object_idx:
                pred_obj.add(pred[j].item())
        score = len(trg_obj.intersection(pred_obj))/max(len(pred_obj), 1)
        scores.append(score)
    reward_shape = (beam_major.shape[0], beam_major.shape[1])
    reward = torch.tensor(scores).view(reward_shape)
    return reward

def get_execute_accuracy(trg, pred, trg_mask, trg_lengths, trg_field):
    '''
    input
        repeated_trg = [beam, batch_size, seq_len]
        beam_major = [beam_size, batch_size, seq_len]
        trg_mask = [beam_size, batch_size, seq_len]
        trg_lengths = [beam, batch]
    returns
        [beam*batch] rewards
    '''
    beam_size, batch_size, trg_seq_len = trg.shape
    pred_seq_len = pred.shape[-1]
    x_flat = pred.contiguous().view(-1, pred_seq_len)
    y_flat = trg.contiguous().view(-1, trg_seq_len)

    x_strings, y_strings = [], []
    for i in range(beam_size*batch_size):
        x_strings.append(trg_field.get_string(x_flat[i]))
        y_strings.append(trg_field.get_string(y_flat[i]))

    scores = []
    for idx, (sys, ref) in enumerate(zip(x_strings, y_strings)):
        executor = Executor(ref)
        score = executor.execute(sys)
        scores.append(score)
    return scores

def get_match_score(repeated_trg, beam_major, trg_mask, trg_lengths):
    '''
    input
        repeated_trg = [beam, batch_size, seq_len]
        beam_major = [beam_size, batch_size, seq_len]
        trg_mask = [beam_size, batch_size, seq_len]
        trg_lengths = [beam, batch]
    returns
        [beam, batch] tensor of rewards
    '''
    reward = ((repeated_trg==beam_major).float())*((~trg_mask).float())
    reward = reward.sum(2)
    bonus = (reward==trg_lengths.float()).float()
    reward = reward/trg_lengths.float() + bonus
    return reward

def compute_nl_rewards(unmasked_beam_words=None, beta=0.0,
                    nl_trg=None, loss_type=None,
                    mc_nl_words=None, mc_nl_out=None, 
                    nl_lengths=None, nl_field=None):
        batch_size, beam_width, nl_seq_len, nl_vocab = mc_nl_out.shape
        nl_trg = nl_trg.unsqueeze(1).expand(-1, beam_width, -1)
        nl_lengths = nl_lengths.unsqueeze(-1).expand(-1, beam_width)

        mc_nl_words, mr_lengths = batch_output(mc_nl_words, nl_field,
                                               device=mc_nl_words.device)

        #NOTE bleu
        #bleu_scores = get_bleu_scores(nl_trg, mc_nl_words, nl_field)
        #bleu_scores = torch.tensor(bleu_scores, device=nl_trg.device)
        #bleu_scores = bleu_scores.view(batch_size, beam_width)
        #return bleu_scores
        #NOTE
        mask_tmp = make_mask(nl_lengths, nl_trg)
        mc_nl_mask = mask_tmp.clone().to(nl_trg.device).float()
        del mask_tmp

        mc_nl_word_probs = mc_nl_out.gather(3, nl_trg.unsqueeze(-1)).squeeze(-1)
        #word_probs (before squeeze) is [batch_size, beam_size, seq_len, 1]

        mc_nl_word_probs = mc_nl_word_probs*mc_nl_mask

        mc_nl_word_probs = mc_nl_word_probs.sum(-1)#[batch, beam], p(x|y)
        mc_nl_word_probs = torch.exp(mc_nl_word_probs)
        return mc_nl_word_probs


def get_example_ltl_score(hyp, mr_field, ltl_args, 
                          env_data, example_id, env_id, item_grid, agent,
                          actions,
                          length_cache=None, score_cache=None,
                          score_type=None, alpha=0.1):
    '''
    get the score for a given example and environment
    '''

    score = 0

    #print("hyp", hyp)
    if not mr_field.vocab.eos_idx in hyp:
        return score

    #print(hyp)
    h_words = mr_field.get_string(hyp, strip_pad=True)
    #print("h_words", h_words)
    #print(h_words)
    #if h_words == "apple apple apple":
    #    print(h_words)
    #    exit()
    infix_words = postfix2infix(h_words, mr_field)
    #print("infix_words", infix_words)
    if len(infix_words)==1:
        #print(item_grid)
        env_primitives = set([x for y in item_grid for x in y])
        env_primitives.remove(0)

        formula = infix_words[0]

        formula = replace_symbols(formula, env_name="Craft")

        formula = fruit_to_craft(formula)

        #idx -> {formula -> {env -> score}}
        #print(score_cache)
        cached_scores = score_cache.get(example_id, {})
        cached_env_scores = cached_scores.get(h_words, {})
        #print(formula)
        if env_id in cached_env_scores:
            return cached_env_scores[env_id]

        ltl_args.formula = formula
        #NOTE height, width hardcode
        env = sample_craft_env_each(ltl_args,7,7,env_data,None)
        #check if command uses non-existent items
        f_split = formula.split(" ")
        cookbook_index = env.cookbook.index.contents
        f_split = filter(lambda x: x in cookbook_index, f_split)
        f_primitives = set([cookbook_index.get(x, -1) for x in f_split])
        is_subset = f_primitives.issubset(env_primitives)
        #print("is_subset", is_subset)
        #print("formula", formula)
        #print("env_primitives", [env.cookbook.index.get(i) for i in env_primitives])
        #print("f_primitives", [env.cookbook.index.get(i) for i in f_primitives])
        if not is_subset:
            cached_env_scores[env_id] = 0
            return score

        ltl_tree = ltl2tree(formula, ltl_args.alphabets, False)
        agent.update_formula(ltl_tree)

        #print(formula)
        env.reset()
        prob, reward, _, _ = run_actions(formula, env, 
                                         agent, ltl_args,
                                         actions=actions,
                                         env_only=score_type=="accept")
        cached_env_scores[env_id] = max(0, reward)
        env.reset()
        #print("reward", reward)
        if reward > 0.9:
            likelihood = [-p for p in prob]
            if score_type=="accept":
                score = 1
            if "log" not in score_type:
                likelihood = [math.exp(p) for p in likelihood]
            if "norm" in score_type:
                likelihood = sum(likelihood)/len(likelihood)
            else:
                if "log" in score_type:
                    likelihood = sum(likelihood)
                else:
                    likelihood = reduce(operator.mul, likelihood, 1)
            #print(score_type, likelihood)
            if "likelihood_only" in score_type:
                score = likelihood
            elif "efficiency" in score_type:
                env.reset()
                gt_length = get_length(formula, env, agent, ltl_args,
                                       actions=actions)
                env.reset()
                sampled_length = get_length(formula, env, agent, ltl_args,
                                            length_cache=length_cache, 
                                            example_idx=example_id,
                                            env_idx=env_id,
                                            num_tracks=10)
                efficiency = math.exp(-alpha*abs(gt_length-sampled_length))
                score = efficiency*likelihood
            cached_env_scores[env_id] = score
        cached_scores[h_words] = cached_env_scores
        score_cache[example_id] = cached_scores
        env.close(); del env
    return score

def ltl_reward_metrics(example_iter, model,
                       trg_field, src_field,
                       length_cache, score_cache, ids,
                       device=None, args=None,
                       ltl_args=None, max_len=None):
    avg_pred_rewards, avg_gt_rewards = 0, 0
    count = 0
    model.eval()
    #print("ltl metrics printing")
    auto_enc = args["model_type"]=="auto_enc"
    score_type = args["score_type"]

    ignore_gen = True
    f1_sum, fsa_sum, accept, executes, count = 0, 0, 0, 0, 0
    nl_trg, nl_trg_lengths = None, None
    mr_out, mr_words, attns = None, None, None
    nl_out, nl_words, nl_out = None, None, None
    for batch_idx, batch in enumerate(example_iter):
        curr_model = model
        if auto_enc:
            nl_trg, nl_trg_lengths = batch["nl_trg"], batch["nl_trg_lengths"]
            #curr_model = model.parser
        #print("auto_enc", auto_enc)
        with torch.no_grad():
            metric_args = args.copy()
            metric_args["decode_mode"]="beam"
            metric_args["beam_width"] = 10
            metric_args["valid_only"]=True#NOTE always valid.
            metric_args["epsilon"]=0
            #print("get_ltl_metrics")
            #print(nl_trg)
            model_out = curr_model.forward(batch["nl_src"], 
                                           batch["mr_trg"],
                                           batch["nl_src_lengths"], 
                                           batch["nl_trg_lengths"],
                                           nl_trg_lengths=nl_trg_lengths,
                                           nl_trg=nl_trg,
                                           teacher_force=False, 
                                           trg_field=trg_field,
                                           src_field=src_field,
                                           args=metric_args,
                                           ignore_gen=ignore_gen,
                                           max_len=max_len)
        if auto_enc and not ignore_gen:
            mr_out, mr_words, _, nl_out, nl_words, _ = model_out
            nl_words, nl_lengths = batch_output(nl_words, src_field, 
                                                device=device)
        else:
            mr_out, mr_words, attns = model_out

        mr_out, mr_words = mr_out.detach(), mr_words.detach()
        mr_words, mr_lengths = batch_output(mr_words, trg_field, device=device)
        #mr_words = mr_words.transpose(0,1) #beam major
        mr_words = mr_words[:,:1,:]#get first beam
        mr_out = mr_out[:,:1,:]
        #print(mr_out.shape)
        #exit()
        #print(mr_words)
        mr_lengths = mr_lengths[:,:1,]
        mr_words = mr_words.to(device)
        planner_path = args["planner_path"]

        rewards = compute_ltl_rewards(mr_words, mr_lengths,
                                      batch["ltl_data"],
                                      trg_field, 
                                      ltl_args,
                                      device=device,
                                      trg=batch["mr_trg_y"],
                                      ids=batch["id"],
                                      length_cache=length_cache,
                                      score_cache=score_cache,
                                      score_type=score_type,
                                      model_path=planner_path )
        avg_pred_rewards += rewards.sum().item()

        mr_words = batch["mr_trg_y"].unsqueeze(1)
        mr_lengths = batch["mr_trg_lengths"].unsqueeze(1)
        rewards = compute_ltl_rewards(mr_words, mr_lengths,
                                      batch["ltl_data"],
                                      trg_field, 
                                      ltl_args,
                                      device=device,
                                      trg=batch["mr_trg_y"],
                                      ids=batch["id"],
                                      length_cache=length_cache,
                                      score_cache=score_cache,
                                      score_type=score_type,
                                      model_path=planner_path )
        avg_gt_rewards += rewards.sum().item()
        count += rewards.shape[0]
    avg_gt_rewards = avg_gt_rewards/count
    avg_pred_rewards = avg_pred_rewards/count
    OUT_DIR = args["out_dir"]
    pickle.dump(length_cache, open(f'{OUT_DIR}/length_cache.pkl', 'wb'))
    pickle.dump(score_cache, open(f'{OUT_DIR}/{score_type}_cache.pkl', 'wb'))
    return avg_pred_rewards, avg_gt_rewards

def compute_ltl_rewards(mr_words, mr_lengths, ltl_data,
                        mr_field, ltl_args, device=None, trg=None, 
                        ids=None, length_cache=None, score_cache=None,
                        score_type=None, alpha=0.1, model_path=None,
                        n_envs=3):
    batch_n, beam_n, seq_n = mr_words.shape

    dummy_formula = "( G ( gold ) )"
    dummy_formula = replace_symbols(dummy_formula, env_name="Craft")
    gt_actions = ltl_data[0]["actions"][0]["steps"][0]
    ltl_args.formula = dummy_formula

    cookbook = Cookbook(ltl_args.recipe_path)
    n_features = cookbook.n_kinds+1
    env_data = get_env_data(ltl_data[0], n_features, 0)

    #print(env_data)

    env = sample_craft_env_each(ltl_args,7,7,env_data,None)#NOTE height, width

    ###TODO
    #envs = make_vec_envs(ltl_args, device, False, env_data)
    #print(envs)
    #exit()
    ###TODO

    ltl_tree = ltl2tree(dummy_formula, ltl_args.alphabets, False)
    ltl_args.observation_space = env.observation_space
    ltl_args.device = mr_words.device
    ltl_args.action_space = env.action_space

    agent = A2C_ACKTRTrainer(ltl_tree, ltl_args.alphabets, ltl_args)
    agent.actor_critic.load_state_dict(torch.load(model_path, map_location=ltl_args.device)[0])#TODO location
    agent.actor_critic.eval()
    ltl_args.update_failed_trans_only = False

    scores = []
    env.reset()
    for i in range(batch_n):
        example_id = ids[i]
        for j in range(beam_n):
            env_scores = []
            hyp = mr_words[i][j]
            n_envs = len(ltl_data[i]["actions"])
            for env_id in range(n_envs):
                #print()
                #print(n_envs)
                #print(env_id)
                actions = ltl_data[i]["actions"][env_id]["steps"][0]
                env_data = get_env_data(ltl_data[i], n_features, env_id)
                #print(env_data)

                item_grid = ltl_data[i]["envs"][env_id]["init_grid"]
                env_score = get_example_ltl_score(hyp, mr_field, ltl_args, 
                                                  env_data, example_id, env_id,
                                                  item_grid, agent, actions,
                                                  length_cache=length_cache, 
                                                  score_cache=score_cache,
                                                  score_type=score_type, alpha=0.1)
                env_scores.append(env_score)
                #print("score", env_score)
            final_score = 0
            if all(env_scores):
                final_score = sum(env_scores)/len(env_scores)
                #print(mr_field.get_string(hyp))
                #print(env_scores)
                #print()
            scores.append(final_score) 
    scores = torch.tensor(scores, device=mr_words.device).float()
    scores = scores.view(batch_n, beam_n)
    del agent
    return scores

def compute_rewards(beam_words, out_lengths, trg, trg_lengths, src_field, 
                    trg_field, src=None, syntax_score=False, 
                    baseline=0.0, object_recog=False, device=None,
                    score_type="simulate"):
    '''
    input
        beam_words = [batch_size, beam_size, seq_len]
        out_lengths = [batch_size] lengths of beam_words
        trg = [batch_size, seq_len]
        trg_lengths = [batch_size, beam_size] lengths of trg
        src = [batch_size, src_seq_len]
    returns
        [batch_size, beam_size] tensor of rewards
    '''
    out_lengths = torch.transpose(out_lengths, 0, 1)#[beam_size, batch_size] 
    batch_size, beam_size, seq_len = beam_words.shape
    trg_lengths = trg_lengths.unsqueeze(0).repeat(beam_size, 1).transpose(0, 1)
    trg_lengths = torch.transpose(trg_lengths, 0, 1)#[beam_size, batch_size] 
    trg_lengths = trg_lengths.to(device)
    beam_major = torch.transpose(beam_words, 0, 1)
    #beam_major is [beam_size, batch_size, seq_len]
    repeated_trg = trg.repeat(beam_size, 1, 1)
    #repeated_trg is [beam_size, batch_size, seq_len]

    repeated_src = src.repeat(beam_size, 1, 1)
    #repeated_trg is [beam_size, batch_size, src_seq_len]

    mask_tmp = make_mask(trg_lengths, beam_major)
    trg_mask = ~mask_tmp.clone().to(device)
    del mask_tmp

    scores = get_execute_score(repeated_trg, beam_major, trg_field, 
                               score_type=score_type)
    scores = torch.tensor(scores).view(beam_size, batch_size).to(device)
    reward = scores.float()

    #baseline = reward.mean(0)#[batch_size] #NOTE toggle
    reward = reward - baseline

    reward = torch.transpose(reward, 0, 1)

    return reward

