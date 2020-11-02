#!/usr/bin/env python
import spot
import pickle
import bcolz
import re
import numpy as np
import torch
import itertools
import argparse
import pickle

from argparse import Namespace
from collections import Counter
from ltl.spot2ba import Automaton
from ltl.envs import make_vec_envs, make_single_env
from ltl.ltl2tree import replace_symbols, ltl2tree
from ltl.worlds.craft_world import sample_craft_env_each
from ltl.worlds.craft_world import Cookbook
from ltl.algo.a2c_acktr import A2C_ACKTRTrainer


def get_args():
    parser = argparse.ArgumentParser(description='RL with LTL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | acktr | sac')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--env_name', default='CharStream',
                        help='environment to train on: CharStream | Craft')
    parser.add_argument('--num_train_ltls', type=int, default=50,
                        help='number of sampled ltl formula for training (default: 50)')
    parser.add_argument('--cuda_deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num_processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='number of forward steps in A2C (default: 10)')
    parser.add_argument('--num_env_steps', type=int, default=5*10e3,
                        help='number of environment steps to train per environment (default: 10e5)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs to go over all formulas (default: 10)')
    parser.add_argument('--log_dir', default='/tmp/ltl-rl/',
                        help='directory to save agent logs (default: /tmp/ltl-rl)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--rnn_size', type=int, default=64,
                        help='dimensions of the RNN hidden layers.')
    parser.add_argument('--rnn_depth', type=int, default=1,
                        help='number of layers in the stacked RNN.')
    parser.add_argument('--output_state_size', type=int, default=32,
                        help='dimensions of the output interpretable state vector.')
    parser.add_argument('--use_gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--prefix_reward_decay', type=float, default=0.03,
                        help='decay of reward if following prefix (default: 0.03)')
    parser.add_argument('--use_proper_time_limits', action='store_true', default=False,
                        help='compute returns taking into account time limits')
    parser.add_argument('--save_dir', default='./models/',
                        help='directory to save agent logs (default: ./models/)')
    parser.add_argument('--save_model_name', default='model.pt',
                        help='name of the saved model (default: model.pt)')
    parser.add_argument('--load_model_name', default='model.pt',
                        help='name of the model to be loaded (default: model.pt)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='train and evaluate baseline model')
    parser.add_argument('--train', action='store_true', default=False,
                        help='in training mode or not')
    parser.add_argument('--load_formula_pickle', action='store_true', default=False,
                        help='train and evaluate baseline model')
    parser.add_argument('--formula_pickle', default='',
                        help='path to load the formulas')
    parser.add_argument('--test_formula_pickle_1', default='',
                        help='path to load the test formulas')
    parser.add_argument('--test_formula_pickle_2', default='',
                        help='path to load the test formulas')
    parser.add_argument('--test_formula_pickle_3', default='',
                        help='path to load the test formulas')
    parser.add_argument('--test_formula_pickle_4', default='',
                        help='path to load the test formulas')
    parser.add_argument('--save_env_data', action='store_true', default=False,
                        help='save environment data')
    parser.add_argument('--load_env_data', action='store_true', default=False,
                        help='load environment data')
    parser.add_argument('--env_data_path', default='./data/env.pickle',
                        help='path to load environment data (default: ./data/env.pickle)')
    parser.add_argument('--load_model', action='store_true', default=False,
                        help='load pretrained model')
    parser.add_argument('--lang_emb', action='store_true', default=False,
                        help='train the language embedding baseline')
    parser.add_argument('--lang_emb_size', type=int, default=32,
                        help='embedding size of the ltl formula (default: 32)')
    parser.add_argument('--image_emb_size', type=int, default=64,
                        help='embedding size of the input image (default: 64)')
    parser.add_argument('--min_epoch', type=int, default=0,
                        help='starting epoch to evaluate (default: 0)')
    parser.add_argument('--min_formula', type=int, default=0,
                        help='starting formula to evaluate (default: 0)')
    parser.add_argument('--gen_formula_only', action='store_true', default=False,
                        help='only generate the training/testing formulas')
    parser.add_argument('--load_eval_train', action='store_true', default=False,
                        help='load the models in the folder first, run eval, and then train from the last one')
    parser.add_argument('--update_failed_trans_only', action='store_true', default=False,
                        help='only update the modules involved in the failed transition')
    # test
    parser.add_argument('--num_test_ltls', type=int, default=50,
                        help='number of sampled ltl formula for testing (default: 50)')
    parser.add_argument('--test_in_domain', action='store_true', default=False,
                        help='test formula that is in training templates')
    parser.add_argument('--test_out_domain', action='store_true', default=False,
                        help='test formula that is not in training templates')
    parser.add_argument('--max_symbol_len', type=int, default=10,
                        help='max number of nodes in formula (default: 10)')
    parser.add_argument('--min_symbol_len', type=int, default=1,
                        help='min number of nodes in formula (default: 1)')
    parser.add_argument('--no_time', action='store_true', default=False,
                        help='evaluate no time dependency')
    args = parser.parse_args([])
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.env_name="Craft"
    args.use_gui=False
    args.is_headless=True#NOTE
    args.target_fps=None#NOTE
    args.update_failed_trans_only=False#NOTE
    args.recipe_path='/storage/czw/rl_parser/ltl/ltl/worlds/craft_recipes_basic.yaml'#NOTE
    args.alphabets = ['C_boundary', 'tree', 'C_tree', 'workbench', 
                      'C_workbench', 'factory', 'C_factory', 'iron', 'C_iron', 
                      'gold', 'C_gold', 'gem', 'C_gem']#NOTE hardcode
    args.rnn_depth = 2#NOTE
    args.num_steps = 15
    return args


class Vocab():
    def __init__(self, sos_token=None, eos_token=None, pad_token=None, 
                 unk_token=None):
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.itos = [unk_token, pad_token, sos_token, eos_token]

        self.stoi = {v:k for k,v in enumerate(self.itos)}

        self.pad_idx = self.stoi[pad_token]
        self.sos_idx = self.stoi[sos_token]
        self.eos_idx = self.stoi[eos_token]
        self.unk_idx = self.stoi[unk_token]

        if sos_token is None or eos_token is None or pad_token is None\
           or unk_token is None:
            self.itos = []
            self.stoi = {}

        self.freqs = Counter()

    def __len__(self):
        return len(self.itos)

    def add_word(self, word):
        self.freqs[word] += 1
        if word not in self.stoi:
            self.stoi[word] = len(self.itos)
            self.itos.append(word)


class Field():
    def __init__(self, name, sos_token=None, eos_token=None, pad_token=None, 
                 unk_token=None):

        self.vocab = Vocab(sos_token=sos_token, eos_token=eos_token, 
                           pad_token=pad_token, unk_token=unk_token)

        #valids maps indices to the set of indices that may follow
        self.valids = None
        self.predicates = None
        self.two_place = None
        self.one_place = None
        self.args = None
        self.nouns = None

    def normalize_string(self, s):
        s = s.strip() #no lower()
        return s

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            if not word.isspace():
                self.vocab.add_word(word)

    def lookup_words(self, x):
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        
        if self.vocab is not None:
            x = [self.vocab.itos[i] for i in x]

        return [str(t) for t in x]

    def get_string(self, x, strip=True, strip_pad=False):
        stripPAD = lambda x: re.sub("{}".format(self.vocab.pad_token), "", x)
        stripEOS = lambda x: re.sub("{}.*".format(self.vocab.eos_token), "", x)
        stripSOS = lambda x: re.sub("{}".format(self.vocab.sos_token), "", x)
        res = " ".join(self.lookup_words(x))
        if strip_pad:
            res = stripPAD(res)
        if strip:
            return stripEOS(stripSOS(res)).strip()
        else:
            return res.strip()

    def check_path(self, word_idxs):
        '''
            sequence needs to have </s> 
            sequence can't have <s> 
            sequence cannot have repeated clauses
        '''
        if self.vocab.eos_idx not in word_idxs:
            #possible that there are two </s>? I don't think so?
            return False
        words = self.get_string(word_idxs, strip_pad=True)
        words = words.split(" ")
        args = []
        predicates = {}
        passed_first_pred = False
        for word in words[::-1]:
            if word.isdigit():
                args.append(word)
            else:
                if predicates.get(word, []) == args:
                    return False
                predicates[word] = args
                args = []
        return True

    def get_valid_idx(self, prev_words):
        '''
            prev_word [beam, seq_len] 
            out:
                list of (list of valid continuation indices) for each sequence
        '''
        res = []
        for word_idxs in prev_words:
            words = [self.vocab.itos[i.item()] for i in word_idxs]
            postfix = self.get_string(word_idxs)
            #print("postfix", postfix)
            #print(postfix != "factory gold (")
            #if postfix == "factory gold (":
            #    print("***************************")
            #    exit()
            assert postfix != "factory gold ("
            assert "<unk>" not in postfix
            infix = postfix2infix(postfix, self, typed=True)
            valids = set()
            add_one_place = False
            if len(words) == 1: #<s>
                valids = self.nouns_idx
            if words[-1] == self.vocab.eos_token or\
               words[-1] == self.vocab.pad_token:
                valids = set([self.vocab.pad_idx])
            elif len(infix) > 0:
                valids = set()
                f_type_0 = infix[0][1]
                valids = valids.union(self.one_place_idx)
                valids = valids.union(self.nouns_idx)
                add_one_place = True
                #if f_type_0[1] == "noun":
                #    valids.remove(self.vocab.stoi[words[-1]])
            if len(infix) > 1:
                f_type_0 = infix[0][1]
                f_type_1 = infix[1][1]
                same_type = f_type_0[1] == f_type_1[1] or\
                            (f_type_0[1] != "time" and f_type_1[1] != "time")

                same_noun = f_type_0[1] == "noun" and\
                            f_type_0[0] == f_type_1[0]
                if same_type and not same_noun:
                    valids = valids.union(self.two_place_idx)
            if add_one_place and f_type_0[0] in self.one_place\
                             and self.vocab.stoi[f_type_0[0]] in valids:
                #no double negation
                valids.remove(self.vocab.stoi[f_type_0[0]])
                #the below is to remove F G F and G F G
                no_paren = infix[0][0].replace("(","").replace(")","")
                prev_tokens = no_paren.strip().split(" ")
                if len(prev_tokens) > 1:
                    prev_symbol = prev_tokens[1]
                    if prev_symbol in ["F", "G"] and\
                        self.vocab.stoi[prev_symbol] in valids:
                        valids.remove(self.vocab.stoi[prev_symbol])
            if add_one_place and "!" in self.vocab.stoi\
                             and self.vocab.stoi["!"] in valids\
                             and f_type_0[1] == "time":
                valids.remove(self.vocab.stoi["!"])
            if len(infix) == 1:
                f_type_0 = infix[0][1]
                if f_type_0[1] == "time":
                    valids.add(self.vocab.eos_idx)
            #print("idx valids:", [self.vocab.itos[x] for x in valids])
            res.append(list(valids))
        return res

    def build_vocab(self, sentences, min_freq=0, trg=False, ltl=False):
        for sentence in sentences:
            sentence = self.normalize_string(sentence)
            self.add_sentence(sentence)

        if trg:
            self.valids = {}
            predicates = set()
            args = set()
            for i in range(4,len(self.vocab)):#skip first four tokens
                word = self.vocab.itos[i]
                if word.isdigit():
                    args.add(i)
                else:
                    predicates.add(i)
            self.predicates = predicates
            self.args = args
            for i in range(len(self.vocab)):
                word = self.vocab.itos[i]
                if word.isdigit():#0, 1, 2
                    self.valids[i] = list(predicates) + list(args) + [self.vocab.eos_idx]
                else:#table, chair, person
                    self.valids[i] = list(args)
            self.valids[self.vocab.unk_idx] = list(args) + list(predicates)
            self.valids[self.vocab.sos_idx] = list(predicates)

            self.one_place = set(["person", "walk", "stand", "bag",
                                   "chair", "two", "apple", "table", 
                                   "car", "book", "pants", "telescope",
                                   "toy", "cars", "three", "hallway",
                                   "floor"])
            self.nouns     = set(["person", "bag",
                                  "chair", "apple", "table", 
                                  "car", "book", "pants", "telescope",
                                  "toy", "cars", "hallway",
                                  "floor"])
            self.two_place = set(["on", "in", "hold", "move", "leave",
                                  "touch", "drop", "wear", "goes_to",
                                  "pick_up", "put_down", "near", "from",
                                  "with", "toward", "across"])
            if ltl:
                self.one_place = set(["F", "G"])
                if "!" in self.vocab.stoi:
                    self.one_place.add("!")
                self.one_place_idx = set([self.vocab.stoi[s] for s in self.one_place])
                #self.nouns     = set(["iron", "gem", "gold", "factory", 
                #                      "workbench", "tree"])
                self.nouns     = set(["apple", "pear", "orange", "flag", 
                                      "house", "tree"])
                #closer_nouns = set(["C_"+x for x in self.nouns])
                #self.nouns = self.nouns.union(closer_nouns)
                self.nouns_idx = set([self.vocab.stoi[s] for s in self.nouns])
                self.two_place = set(["&", "|"])
                self.two_place_idx = set([self.vocab.stoi[s] for s in self.two_place])
                
    def to_idxs(self, sentence):
        words = self.normalize_string(sentence).split(' ') 
        unk_idx = 0
        if self.vocab.unk_token is not None:
            unk_idx = self.vocab.stoi[self.vocab.unk_token]
        return [self.vocab.stoi.get(w, unk_idx) for w in words]

def make_mask(lengths, x):
    '''
        lengths = [batch_size, beam_size]
        x = [batch_size, beam_size, seq_len]
    '''
    seq_len = x.shape[-1]
    #print(lengths[0])
    flat_lengths = lengths.contiguous().view(-1, 1).long().cpu()
    mask = torch.arange(0, seq_len).unsqueeze(0).repeat(flat_lengths.size(0), 1)
    #print(mask)
    #print(flat_lengths)
    mask = mask < flat_lengths
    mask = mask.view(x.shape)
    return mask

def batch_output(beam_words, trg_field, device=None):
    '''
    input
        beam_words = [batch_size, beam_size, seq_len]
    return
        padded = [batch_size, beam_size, seq_len]
        tensor_lengths = [batch_size, beam_size]
    '''
    batch_size, beam_size, seq_len  = beam_words.shape
    pad_index = trg_field.vocab.stoi[trg_field.vocab.pad_token]
    eos_index = trg_field.vocab.stoi[trg_field.vocab.eos_token]
    lengths = []

    #NOTE: did some stuff to ensure that padding is added, but every sentence is not capped with an EOS
    beam_words_temp = beam_words.detach().clone().type_as(beam_words).to(device)
    beam_words_temp[:,:, seq_len-1] = eos_index
    
    beam_words = beam_words.detach().clone().view(batch_size*beam_size, seq_len)
    beam_words_temp = beam_words_temp.view(batch_size*beam_size, seq_len)
    new_lengths = [torch.where(x==eos_index)[0][0].item() + 1 for x in beam_words_temp]
 
    mask = []
    for length in new_lengths:
        mask.append([[0]*length + [1]*(seq_len - length)])
    
    tensor_lengths = torch.LongTensor(new_lengths)
    tensor_lengths = tensor_lengths.view(batch_size, beam_size)
    tensor_lengths = tensor_lengths.to(device).type(torch.float)
    tensor_mask = torch.Tensor(mask).ne(0).to(device)

    beam_words.masked_fill_(tensor_mask.squeeze()==1, pad_index)
    beam_words = beam_words.view(batch_size, beam_size, seq_len)
    return beam_words, tensor_lengths

def gen_forward_pass(mr_outs, mr_words, mr_attns, model,
                     nl_field=None, mr_field=None, nl_trg=None, nl_trg_lengths=None,
                     require_eos=True, ignore_gen=None, args=None, max_len=None):
    gen_args = args.copy()
    gen_args["valid_only"] = False
    inter_outs, inter_words, inter_attns = mr_outs, mr_words, mr_attns
    #print(inter_words.shape)

    src_device = inter_words.device
    inter_words = inter_words.detach()
    pad_inter_words, inter_lengths = batch_output(inter_words, 
                                                  mr_field, src_device)

    batch_size, beam_width, in_seq_len, vocab_size = inter_outs.shape
    out_seq_len = nl_trg.shape[-1]
    inter_words = inter_words.view(-1, in_seq_len)#NOTE: not feeding the pad
    inter_lengths = inter_lengths.view(-1)
    inter_lengths = inter_lengths.long()
    nl_trg = nl_trg.unsqueeze(1)
    nl_trg = nl_trg.expand(-1, beam_width, -1)
    nl_trg = nl_trg.contiguous().view(-1, out_seq_len)

    #teacher forcing!
    gen_beam_width=1#TODO
    gen_args = args.copy()
    gen_args["beam_width"] = gen_beam_width
    gen_args["decode_mode"] = "beam"
    gen_args["epsilon"] = 0
    gen_args["valid_only"] = False
    #print(inter_words.shape)
    teacher_force = gen_args["teacher_force"]
    outs, words, attns = model.generator.forward(inter_words, #src
                                                 nl_trg, #trg
                                                 inter_lengths,
                                                 nl_trg_lengths, 
                                                 teacher_force=teacher_force,#NOTE NB
                                                 max_len=max_len,
                                                 trg_field=nl_field,
                                                 require_eos=False,
                                                 args=gen_args)
    #print(words)
    #for row in words:
    #    print(nl_field.lookup_words(row[0]))
    out_seq_len = words.shape[-1]
    outs = outs.view(batch_size, beam_width, out_seq_len, -1)
    words = words.view(batch_size, beam_width, out_seq_len)
    return outs, words, attns


def get_string_rep(example_iter, model, n=2,
                   sos_index=1, 
                   src_eos_index=None, 
                   trg_eos_index=None, 
                   src_field=None, trg_field=None,
                   device=None, require_eos=True,
                   args=None, max_len=None):

    auto_enc = args["model_type"]=="auto_enc"
    model.eval()
    count = 0
    src_nls = []
    trg_mrs = []
    pred_mrs = []
    pred_nls = []
    #print("train_examples printing")
    nl_trg, nl_trg_lengths = None, None
    mr_out, mr_words, attns = None, None, None
    nl_out, nl_words, nl_out = None, None, None

    for batch_idx, batch in enumerate(example_iter):
        curr_model = model
        if auto_enc:
            nl_trg, nl_trg_lengths = batch["nl_trg"], batch["nl_trg_lengths"]
        with torch.no_grad():
            string_args = args.copy()
            string_args["beam_width"] = 10
            string_args["decode_mode"] = "beam"
            string_args["epsilon"] = 0
            model_out = curr_model.forward(batch["nl_src"], 
                                      batch["mr_trg"],
                                      batch["nl_src_lengths"], 
                                      batch["mr_trg_lengths"],
                                      teacher_force=False,
                                      nl_trg=nl_trg,
                                      nl_trg_lengths=nl_trg_lengths,
                                      trg_field=trg_field,
                                      src_field=src_field,
                                      require_eos=require_eos,
                                      args=string_args, 
                                      max_len=max_len)

        if auto_enc:
            mr_out, mr_words, _, nl_out, nl_words, _ = model_out
            nl_words, nl_lengths = batch_output(nl_words, src_field, 
                                                device=device)
        else:
            mr_out, mr_words, attns = model_out

        if auto_enc:
            #NOTE: problem: the best_outs, best_words
            mr_out, mr_words, _, nl_out, nl_words, _ = model_out
            nl_words, nl_lengths = batch_output(nl_words, src_field, 
                                                device=device)
            nl_words = nl_words[:,0,:] #get first beam result#NOTE the beam is always 1
        else:
            mr_out, mr_words, attns = model_out

        mr_out, mr_words = mr_out.detach(), mr_words.detach()
        mr_words, out_lengths = batch_output(mr_words, trg_field, device=device)#TODO check if need this
        #beam words is [batch_size, beam_size, seq_len]
        mr_words = mr_words[:,0,:] #get first beam result
        #print(beam_words)
        #print("string_rep")
        #print(beam_words)
        for example_idx in range(batch["nl_src"].shape[0]):
            nl_src = src_field.get_string(batch["nl_src"][example_idx])
            mr_trg = trg_field.get_string(batch["mr_trg"][example_idx], 
                                 strip_pad=True)

            mr_pred = trg_field.get_string(mr_words[example_idx])
            src_nls.append(nl_src)
            trg_mrs.append(mr_trg)
            pred_mrs.append(mr_pred)

            if auto_enc:
                nl_pred = src_field.get_string(nl_words[example_idx])
                pred_nls.append(nl_pred) 
            count += 1
            if count >= n:
                break
        if count >= n:
            break
    if auto_enc:
        return src_nls, trg_mrs, pred_mrs, pred_nls
    else:
        return src_nls, trg_mrs, pred_mrs

class Executor():
    '''
        An order-invariant representation of a logical form
    '''
    def __init__(self, pred, trg_field):
        self.predicates = {} 
        self.num_predicates = 0
        self.nouns = set()
        self.vars = set()
        self.string = pred
        self.trg_field = trg_field

        args = []
        for word in pred.split(" ")[::-1]:
            if word.isdigit():
                args.append(word)
                self.vars.add(word)
            else:
                self.num_predicates += 1
                if word in self.predicates:
                    self.predicates[word].append(args)
                else:
                    self.predicates[word] = [args]
                if word in trg_field.nouns:
                    self.nouns.update(args)
                args = []

    def compare_execution(self, gold, sys_idx, trg_field, score_type="simulate"):
        '''
            Compare the predicted meaning representation to the reference
            meaning representation

            Input:
                gold = executor for the reference 
                sys_idx = idxs of the predicted sequence
        '''
        args = []
        num = 0
        if not self.vars.issubset(self.nouns) or not trg_field.check_path(sys_idx):
            return 0
        if score_type=="simulate":
            return self.compute_f1(gold)
        elif score_type=="simulate_subset_pred":
            return self.compute_subset(gold, trg_field, object_only=False)
        elif score_type=="simulate_subset_obj":
            return self.compute_subset(gold, trg_field, object_only=True)
        return None

    def compute_f1(self, other):
        '''
            Compute f1 between the set of predicted clauses and
            the reference set of clauses.
        '''
        other_preds = set(other.predicates.keys())
        my_preds = set(self.predicates.keys())
        intersect = other_preds.intersection(my_preds)
        num = 0
        for key in intersect:
            for arg_list in other.predicates[key]:
                if arg_list in self.predicates[key]:
                    num += 1
        precision = num/max(other.num_predicates, 1)
        recall = num/max(self.num_predicates, 1)
        denom = precision + recall
        denom = 1 if denom == 0 else denom
        f1 = 2*(precision*recall)/(denom)
        return f1

    def compute_subset(self, other, trg_field, object_only=False):
        other_preds = set(other.predicates.keys())
        my_preds = set(self.predicates.keys())

        my_nouns = set(filter(lambda x: x in trg_field.nouns, my_preds))
        other_nouns = set(filter(lambda x: x in trg_field.nouns, other_preds))
        if object_only:
            subset = my_nouns.issubset(other_nouns)
        else:
            subset = my_preds.issubset(other_preds)
            #NOTE todo
            if subset:
                subset = len(my_preds.intersection(other_preds))
        #subset = True
        #for key in intersect:
        #    for arg_list in other.predicates[key]:
        #        if arg_list in self.predicates[key]:
        #            if object_only and self.predicates in trg_field.nouns:
        #                subset = True
        #            elif not object_only:
        #                subset = True
        return float(subset)

    def get_string_variants(self, ref):
        self_vars = list(self.vars)
        ref_vars = list(ref.vars)

        if len(self_vars) > len(ref_vars):
            combos = itertools.permutations(self_vars, len(ref_vars))
        else:
            combos = itertools.permutations(ref_vars, len(self_vars))

        combo_dicts = []
        for c in combos:
            combo_dict = {}
            for i in range(len(c)):
                if len(self_vars) > len(ref_vars):
                    combo_dict[c[i]] = ref_vars[i]
                else:
                    combo_dict[self_vars[i]] = c[i]
            combo_dicts.append(combo_dict)

        variants = []
        words = self.string.split(" ")
        for combo_dict in combo_dicts:
            v = []
            for w in words:
                if w.isdigit() and w not in combo_dict:
                    for i in range(4):#fill with something that won't conflict
                        if i not in combo_dict.values():
                            combo_dict[w] = str(i)
                v.append(combo_dict.get(w, w))
            variants.append(" ".join(v))
        return variants

def print_scores(trg, pred, trg_field, scores):
    '''
    input
        repeated_trg = [beam, batch_size, seq_len]
        pred = [beam_size, batch_size, seq_len]
    returns
        [beam*batch] rewards
    '''
    beam_size, batch_size, trg_seq_len = trg.shape
    pred_seq_len = pred.shape[-1]
    score_tensor = torch.tensor(scores)
    score_tensor = score_tensor.view(beam_size, batch_size)
    score_tensor = score_tensor.transpose(0,1)
    x_batch = pred.transpose(0, 1)
    y_batch = trg.transpose(0, 1)
    for i in range(batch_size):
        x_strings = []
        y_strings = []
        for j in range(beam_size):
            pred = trg_field.get_string(x_batch[i,j])
            trg = trg_field.get_string(y_batch[i,j])
            x_strings.append(pred)
            y_strings.append(trg)
        scores = score_tensor[i].tolist()
        zipped = zip(scores, x_strings, x_batch.tolist()[i])
        print("trg:")
        print(y_strings[0])
        print("pred:")
        print(list(sorted(zipped, key=lambda x: x[0], reverse=True))[:5])

def get_max_execute_score(ref, sys, sys_idx, trg_field, score_type="simulate"):
    if not trg_field.check_path(sys_idx):
        return 0
    if trg_field.vocab.eos_idx not in sys_idx:
        return 0
    if "" in sys.split(" "): #a hack, because <s> sometimes gets stripped
        return 0
    ref_exec = Executor(ref, trg_field)
    sys_exec = Executor(sys, trg_field)
    string_variants = sys_exec.get_string_variants(ref_exec)
    max_score = 0
    for variant in string_variants:
        var_exec = Executor(variant, trg_field)
        var_idx = [trg_field.vocab.stoi[x] for x in variant.split(" ")]
        var_idx.append(trg_field.vocab.eos_idx)
        f1 = var_exec.compare_execution(ref_exec, var_idx, trg_field,
                                        score_type=score_type)
        max_score = max(f1, max_score)
    return max_score

def get_execute_score(trg, pred, trg_field, score_type="simulate"):
    '''
    input
        repeated_trg = [beam, batch_size, seq_len]
        pred = [beam_size, batch_size, seq_len]
    returns
        [beam*batch] rewards
    '''
    beam_size, batch_size, trg_seq_len = trg.shape
    pred_seq_len = pred.shape[-1]
    x_flat = pred.contiguous().view(-1, pred_seq_len)
    y_flat = trg.contiguous().view(-1, trg_seq_len)

    x_strings, y_strings, x_ints = [], [], []
    for i in range(beam_size*batch_size):
        x_strings.append(trg_field.get_string(x_flat[i]))
        y_strings.append(trg_field.get_string(y_flat[i]))
        x_ints.append(x_flat[i].tolist())

    scores = []
    for idx, (sys, ref, sys_idx) in enumerate(zip(x_strings, y_strings, x_ints)):
        score = 0
        if score_type in ["simulate", "simulate_subset_pred", 
                          "simulate_subset_obj"]:
            score = get_max_execute_score(ref, sys, sys_idx, trg_field,
                                          score_type=score_type)
        else:
            print("unknown score type")
        scores.append(score)
    #print_scores(trg, pred, trg_field, scores)
    return scores

craft_to_fruit_d = {"gem": "apple",
                    "gold": "orange",
                    "iron": "pear",
                    "factory": "flag",
                    "tree": "tree",
                    "workbench": "house"}
fruit_to_craft_d = {x:y for y, x in craft_to_fruit_d.items()}

def craft_to_fruit(f):
    '''
    gem -> apple
    gold -> orange
    iron -> pear

    factory -> flag
    tree -> tree
    workbench -> house
    '''
    res = f
    for craft in craft_to_fruit_d:
        res = res.replace(craft, craft_to_fruit_d[craft])
    return res

def fruit_to_craft(f):
    res = f
    for fruit in fruit_to_craft_d:
        res = res.replace(fruit, fruit_to_craft_d[fruit])
    return res

def print_timing(msg, time, verbose=1):
    if verbose > 0:
        print(msg, time)

def fsa_match(hyp, trg, mr_field):
    hyp = hyp[:,0,:]#take first beam
    res = 0
    for i in range(hyp.shape[0]):
        h, t = hyp[i], trg[i]
        h_words = mr_field.get_string(h, strip_pad=True)
        t_words = mr_field.get_string(t, strip_pad=True)
        h_words = postfix2infix(h_words, mr_field)
        t_words = postfix2infix(t_words, mr_field)[0]
        if len(h_words)>0:
            h_words = h_words[0]
            equiv = spot.are_equivalent(h_words, t_words)
            res += float(equiv)
    return res

def one_hot(grid, n_features):
    width, height = grid.shape
    one_hot = np.zeros((width*height, n_features))
    flat_grid = grid.flatten()
    one_hot[np.arange(width*height), flat_grid] = 1
    for row in one_hot:
        assert row.sum() == 1
    one_hot[:,0] = 0 #0 encoded as 0
    one_hot = one_hot.reshape((width, height, n_features))
    return one_hot

def get_env_data(ltl_data, n_features, env_id):
    init_pos  = ltl_data["envs"][env_id]["init_pos"]
    init_dir  = ltl_data["envs"][env_id]["init_dir"]
    init_grid = ltl_data["envs"][env_id]["init_grid"]
    #print(init_grid)
    init_grid = np.array(init_grid).astype(int)
    init_grid = one_hot(init_grid, n_features)

    env_data = init_grid, init_pos, init_dir
    return env_data

def run_actions(formula, env, agent, ltl_args, actions=None, num_break_steps=None,
                sample_length=False, mode=False, env_only=False):
    ltl_args.formula = formula
    assert num_break_steps is None or sample_length
    if num_break_steps is None:
        num_break_steps = ltl_args.num_steps 

    #print("ltl args num_break_steps", num_break_steps)
    prob, sampled_actions, last_reward = [], [], -1
    first_accept = 0

    if env_only:
        for step in range(ltl_args.num_steps):
            action = actions[step]
            obs, reward, done, infos = env.step(action)
            last_reward = reward
            sampled_actions.append(action)
            if reward < 0.9:
                first_accept = step
            if done:
                break
        return prob, last_reward, sampled_actions, first_accept

    first_accept = 0
    with torch.no_grad():
        agent.actor_critic.reset()
        obs = env.reset()
        done = False;
        for step in range(ltl_args.num_steps):
            test_obs = []
            for _, s in obs.items():
                test_obs.append(torch.FloatTensor(s))
                test_obs[-1] = test_obs[-1].to(ltl_args.device)
            test_obs = tuple(test_obs)
            mask = torch.FloatTensor([1.0])
            mask = mask.to(ltl_args.device)
            if actions is not None:
                action = actions[step]
                action = torch.LongTensor([action])
                action = action.to(ltl_args.device)#possible memory leak
                _, _, log_prob = agent.actor_critic.evaluate_actions(test_obs,
                                                             mask, action)
            else:
                _, action, log_prob = agent.actor_critic.act(test_obs, mask,
                                                         deterministic=mode, 
                                                         no_hidden=ltl_args.no_time)
            prob.append(log_prob.item())
            obs, reward, done, infos = env.step(action[0].item())
            #obs, reward, done, infos = env.step(4)
            #print("reward", reward)
            last_reward = reward
            sampled_actions.append(action[0].item())
            if reward < 0.9:
                first_accept = step
            if done:
                break
            if sample_length and first_accept > num_break_steps:
                last_reward = -1
                break
    return prob, last_reward, sampled_actions, first_accept

def get_length(formula, env, agent, ltl_args, 
               length_cache=None, example_idx=None, env_idx=None, actions=None,
               num_tracks=None):
    #cache {example_idx -> {formula -> {env_id -> lengths}}}
    first_accept = ltl_args.num_steps
    if actions is not None:
        prob, reward, actions, length = run_actions(formula, env, 
                                                    agent, ltl_args,
                                                    actions=actions)
        if reward > 0.9:
            first_accept = length
    else:
        lengths = length_cache.get(example_idx, {})
        env_lengths = lengths.get(formula, {})
        if env_idx in env_lengths:
            return env_lengths[env_idx]
        env.reset()
        _, reward, _, length = run_actions(formula, env, 
                                            agent, ltl_args,
                                            num_break_steps=first_accept,
                                            sample_length=True,
                                            mode=True)
        if reward > 0.9:
            first_accept = min(first_accept, length)
        for i in range(num_tracks):
            env.reset()
            _, reward, _, length = run_actions(formula, env, 
                                              agent, ltl_args,
                                              num_break_steps=first_accept,
                                              sample_length=True,
                                              mode=False)
            if reward > 0.9:
                first_accept = min(first_accept, length)
                env_lengths[env_idx] = first_accept
                lengths[formula] = env_lengths
                length_cache[example_idx] = lengths
                if first_accept == 0:
                    return first_accept
    return first_accept

def execute(hyp, trg, ltl_data, ltl_args, mr_field, args):
    '''
    does it elicit the right execution
    '''
    count = 0
    hyp = hyp[:,0,:]
    for i in range(hyp.shape[0]):
        h = hyp[i]
        t = trg[i]
        h_words = mr_field.get_string(h, strip_pad=True)
        t_words = mr_field.get_string(t, strip_pad=True)
        #print(h_words)
        h_words = postfix2infix(h_words, mr_field)
        t_words = postfix2infix(t_words, mr_field)
        if len(h_words)>0:
            formula = h_words[0]
            t_formula = t_words[0]
            formula = replace_symbols(formula, env_name="Craft")
            t_formula = replace_symbols(t_formula, env_name="Craft")
            formula = fruit_to_craft(formula)
            t_formula = fruit_to_craft(t_formula)
            ltl_args.formula=formula
            ltl_args.update_failed_trans_only = False
            ltl_args.num_steps = 20#TODO hardcode
            ltl_args.prefix_reward_decay=0.8

            failure = False
            n_envs = len(ltl_data[i]["actions"])

            #print(n_envs)

            for env_id in range(n_envs):
                gt_actions = ltl_data[i]["actions"][env_id]["steps"][0]
                cookbook = Cookbook(ltl_args.recipe_path)
                n_features = cookbook.n_kinds+1
                env_data = get_env_data(ltl_data[i], n_features, env_id)
                env = sample_craft_env_each(ltl_args,7,7,env_data,None)#TODO height width
                env.reset()

                ltl_args.observation_space = env.observation_space
                ltl_args.action_space = env.action_space
                ltl_args.device = hyp.device
                ltl_tree = ltl2tree(formula, ltl_args.alphabets, False)
                
                agent = A2C_ACKTRTrainer(ltl_tree, ltl_args.alphabets, ltl_args)
                model_path = args["planner_path"]
                agent.actor_critic.load_state_dict(torch.load(model_path, 
                                                  map_location=ltl_args.device)[0])#TODO location
                agent.actor_critic.eval()
                ltl_args.update_failed_trans_only = False
                agent.update_formula(ltl_tree)

                prob, reward, actions, length = run_actions(formula, env, 
                                                            agent, ltl_args,
                                                            actions=None,
                                                            mode=True)

                #Now, check if the target accepts
                #ltl_args.formula="asdf"#TODO
                #print(formula)
                #print(reward)
                ltl_args.formula=t_formula
                env = sample_craft_env_each(ltl_args,7,7,env_data,None)#TODO height width
                env.reset()

                reward = 0
                agent = None
                if len(actions) == ltl_args.num_steps:
                    prob, reward, actions, length = run_actions(t_formula, env, 
                                                                agent, 
                                                                ltl_args,
                                                                actions=actions,
                                                                mode=True,
                                                                env_only=True)

                #print(t_formula)
                #print(reward)
                #print()
                #print("reward", reward)
                #print("actions", actions)
#                for action in gt_actions:
#                    obs, reward, done, _ = env.step(action)
#                    #print(reward, done)
#                    if done:
#                        if reward < 0.9:
#                            failure = True
#                        break
#            if not failure:
                #count += reward
                count += max(reward, 0)/(n_envs)
    return count

def action_acceptance(batch_actions, ltl_data, ltl_args, mr_field):
    '''
    is the ground truth accepted by the hypothesis?
    TODO: look into using run_actions()
    '''
    count = 0
    for i in range(batch_actions.shape[0]):
        formula = ltl_data[i]["rewritten_formula"]
        formula = fruit_to_craft(formula)
        ltl_args.formula = formula
        ltl_args.update_failed_trans_only = False
        ltl_args.num_steps = 20#TODO hardcode
        ltl_args.prefix_reward_decay=0.8

        failure = False
        cookbook = Cookbook(ltl_args.recipe_path)
        n_features = cookbook.n_kinds+1
        env_data = get_env_data(ltl_data[i], n_features, 0)#NOTE 0 hardcode
        env = sample_craft_env_each(ltl_args,7,7,env_data,None)#TODO height width
        env.reset()
        actions = batch_actions[i]
        for action in actions:
            obs, reward, done, _ = env.step(action)
            #print(reward, done)
            if done:
                if reward < 0.9:
                    failure = True
                break
        if not failure:
            count += 1
    return count

def acceptance(hyp, ltl_data, ltl_args, mr_field):
    '''
    is the ground truth accepted by the hypothesis?
    '''
    count = 0
    hyp = hyp[:,0,:]
    for i in range(hyp.shape[0]):
        h = hyp[i]
        h_words = mr_field.get_string(h, strip_pad=True)
        h_words = postfix2infix(h_words, mr_field)
        if len(h_words)>0:
            formula = h_words[0]
            formula = replace_symbols(formula, env_name="Craft")
            formula = fruit_to_craft(formula)
            ltl_args.formula=formula
            ltl_args.update_failed_trans_only = False
            ltl_args.num_steps = 20#TODO hardcode
            ltl_args.prefix_reward_decay=0.8

            failure = False
            n_envs = len(ltl_data[i]["actions"])
            for env_id in range(n_envs):
                gt_actions = ltl_data[i]["actions"][env_id]["steps"][0]
                cookbook = Cookbook(ltl_args.recipe_path)
                n_features = cookbook.n_kinds+1
                env_data = get_env_data(ltl_data[i], n_features, env_id)
                env = sample_craft_env_each(ltl_args,7,7,env_data,None)#TODO height width
                env.reset()
                for action in gt_actions:
                    obs, reward, done, _ = env.step(action)
                    #print(reward, done)
                    if done:
                        if reward < 0.9:
                            failure = True
                        break
            if not failure:
                count += 1
    return count

def pad_tensor(x, length=None, pad_idx=1):
    '''
    pad last dimension to be length
    '''
    if length - x.shape[-1] <= 0:
        return x
    new_shape = (*x.shape[:-1], length - x.shape[-1])
    padding = x.new_full(new_shape, pad_idx)
    pad_x = torch.cat((x, padding), dim=len(x.shape)-1)
    return pad_x

def string_match(hyp, trg, hyp_lengths, trg_lengths, trg_field):
    hyp_lengths = hyp_lengths.to(hyp.device)
    trg_lengths = trg_lengths.to(hyp.device)

    h_length = hyp.shape[-1]
    t_length = trg.shape[-1]
    trg_pad = pad_tensor(trg, length=h_length, pad_idx=1)#NOTE hardcode
    hyp_pad = pad_tensor(hyp, length=t_length, pad_idx=1)

    mask = make_mask(hyp_lengths, hyp_pad)
    mask = mask.to(hyp.device)

    hyp_pad = hyp_pad[:,0]
    hyp_lengths = hyp_lengths[:,0]
    mask = mask[:,0]
    for i in range(hyp_pad.shape[0]):#workaround for pad_idx
        if not (hyp_pad[i]==trg_field.vocab.eos_idx).any():
            mask[i] = False
    correct = (hyp_pad==trg_pad)*mask
    correct = correct.sum(-1).float()

    prec = correct/hyp_lengths.float()
    recall = correct/trg_lengths.float()

    f1 = 2*(prec*recall)/(prec+recall)
    nan_mask = f1!=f1
    f1.masked_fill_(nan_mask, 0)
    return f1

def get_action_metrics(example_iter, model, trg_field, src_field,
                       n=-1, device=None, args=None, ltl_args=None,
                       max_len=None, get_execute=False):
    model.eval()
    accept = 0
    nl_trg, nl_trg_lengths = None, None
    mr_out, mr_words, attns = None, None, None
    nl_out, nl_words, nl_out = None, None, None
    count = 0
    for batch_idx, batch in enumerate(example_iter):
        with torch.no_grad():
            metric_args = args.copy()
            model_out = model.forward(batch["nl_src"], 
                                      batch["acts_trg"],
                                      batch["nl_src_lengths"], 
                                      batch["acts_trg_lengths"],
                                      batch["ltl_data"],
                                      teacher_force=False,
                                      nl_trg=nl_trg,
                                      nl_trg_lengths=nl_trg_lengths,
                                      trg_field=trg_field,
                                      src_field=src_field,
                                      require_eos=False,
                                      ignore_gen=True,
                                      args=args, max_len=max_len)

        out, actions = model_out
        out, actions = out.detach(), actions.detach()

        #print(actions.shape)
        accept += action_acceptance(actions, batch["ltl_data"], 
                                    ltl_args, trg_field)
        count += actions.shape[0]
    accept = accept/count
    return accept


def get_ltl_metrics(example_iter, model, trg_field, src_field,
                    n=-1, device=None, args=None, ltl_args=None,
                    max_len=None, get_execute=False):
    model.eval()
    #print("ltl metrics printing")
    auto_enc = args["model_type"]=="auto_enc"
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

        f1_sum += string_match(mr_words, batch["mr_trg_y"], mr_lengths,
                               batch["mr_trg_lengths"], trg_field).sum()
        fsa_sum += fsa_match(mr_words, batch["mr_trg_y"], trg_field)
        accept += acceptance(mr_words, batch["ltl_data"], ltl_args, trg_field)
        if get_execute:
            executes += execute(mr_words, batch["mr_trg_y"], batch["ltl_data"], 
                                ltl_args, trg_field, args)
        count += mr_words.shape[0]
    f1 = f1_sum/count
    f1 = f1.item()
    fsa_score = fsa_sum/count
    accept = accept/count
    executes = executes/count
    return f1, fsa_score, accept, executes

def get_metrics(example_iter, model, trg_field, src_field,
                n=-1, device=None, score_type="simulate", args=None):
    model.eval()
    auto_enc = args["model_type"]=="auto_enc"
    nl_trg, nl_trg_lengths = None, None
    auto_enc = args["model_type"]=="auto_enc"
    correct, nl_correct, total = 0, 0, 0
    for batch_idx, batch in enumerate(example_iter):
        if auto_enc:
            nl_trg, nl_trg_lengths = batch["nl_trg"], batch["nl_trg_lengths"]
        with torch.no_grad():
            metric_args = args.copy()
            metric_args["decode_mode"]="beam"
            metric_args["valid_only"]=True
            model_out = model.forward(batch["nl_src"], 
                                      batch["mr_trg"],
                                      batch["nl_src_lengths"], 
                                      batch["nl_trg_lengths"],
                                      nl_trg=nl_trg,
                                      nl_trg_lengths=nl_trg_lengths,
                                      teacher_force=False, 
                                      trg_field=trg_field,
                                      src_field=src_field,
                                      args=metric_args)
        if auto_enc:
            beam_out, beam_words, attns, nl_out, nl_words, nl_out = model_out
            nl_words, nl_lengths = batch_output(nl_words, src_field, 
                                                device=device)
        else:
            beam_out, beam_words = model_out[0], model_out[1]
        beam_out, beam_words = beam_out.detach(), beam_words.detach()
        beam_words, out_lengths = batch_output(beam_words, trg_field, device=device)
        beam_words = beam_words.transpose(0,1) #beam major
        beam_words = beam_words[:1,:,:]#get first beam
        beam_words = beam_words.to(device)
        #doesn't matter [beam, batch] or [batch, beam]
        scores = get_execute_score(batch["mr_trg_y"].unsqueeze(0), 
                                   beam_words, trg_field, score_type=score_type)
        scores = torch.tensor(scores).float()
        correct += scores.sum().detach()
        total += batch["nseqs"]

        if auto_enc:
            nl_num = (nl_words[:,0]==batch["nl_trg_y"]).sum(1).to(device).float()#correct
            nl_correct += (nl_num/nl_words.shape[-1]).sum()
        if n > 0 and batch_idx > n:
            break
    return correct/total, nl_correct/total

def compute_execute_score(example_iter, model, trg_field, src_field,
                          n=-1, device=None, args=None):
    auto_enc = args["model_type"]=="auto_enc"
    score_type = args["score_type"]
    model.eval()
    correct, total = 0, 0
    nl_trg, nl_trg_lengths = None, None
    auto_enc = args["model_type"]=="auto_enc"
    if auto_enc:
        nl_trg, nl_trg_lengths = batch["nl_trg"], batch["nl_trg_lengths"]
    for batch_idx, batch in enumerate(example_iter):
        with torch.no_grad():
            metric_args = args.copy()
            metric_args["decode_mode"]="beam"
            metric_args["valid_only"]=True
            model_out = model.forward(batch["nl_src"], 
                                      batch["mr_trg"],
                                      batch["nl_src_lengths"], 
                                      batch["mr_trg_lengths"],
                                      nl_trg=nl_trg,
                                      nl_trg_lengths=nl_trg_lengths,
                                      teacher_force=False, 
                                      trg_field=trg_field,
                                      src_field=src_field,
                                      args=metric_args)
        if auto_enc:
            beam_out, beam_words, attns, nl_out, nl_words, nl_out = model_out
        else:
            beam_out, beam_words = model_out[0], model_out[1]
        beam_out, beam_words = beam_out.detach(), beam_words.detach()
        beam_words, out_lengths = batch_output(beam_words, trg_field, device=device)
        beam_words = beam_words.transpose(0,1) #beam major
        beam_words = beam_words[:1,:,:]#get first beam
        beam_words = beam_words.to(device)
        #doesn't matter [beam, batch] or [batch, beam]
        scores = get_execute_score(batch["mr_trg_y"].unsqueeze(0), 
                                   beam_words, trg_field, score_type=score_type)
        scores = torch.tensor(scores).float()
        correct += scores.sum().detach()
        total += batch["nseqs"]
        if n > 0 and batch_idx > n:
            break
    return correct/total

def compute_accuracy(example_iter, model, trg_field, n=-1, beam_width=None,
                     valid_only=False, device=None, args=None):
    '''
    calculates precision (how many of the predicted tokens are correct)
    '''
    model.eval()
    correct, total = 0, 0
    for batch_idx, batch in enumerate(example_iter):
        with torch.no_grad():
            acc_args=args.copy()
            acc_args["decode_mode"]="beam"
            beam_out, beam_words, _ = model.forward(batch["nl_src"], 
                                                    batch["mr_trg"],
                                                    batch["nl_src_lengths"], 
                                                    batch["nl_trg_lengths"],
                                                    teacher_force=False,
                                                    trg_field=trg_field,
                                                    args=acc_args)

        beam_out, beam_words = beam_out.detach(), beam_words.detach()
        beam_words, out_lengths = batch_output(beam_words, trg_field, device=device)
        beam_words = beam_words[:,0,:]#get first beam
        out_lengths = out_lengths[:,0]
        mask = batch["mr_trg_y"].eq(1)#TODO dangerous. better to use length
        beam_words = beam_words.to(device)
        correct += (beam_words==batch["mr_trg_y"]).masked_fill_(mask, False).sum().item()
        total += sum(out_lengths)
        if n > 0 and batch_idx > n:
            break
    return correct/total

def print_action_examples(example_iter, model, n=2, max_len=100, 
                           sos_index=1, src_eos_index=None, 
                           trg_eos_index=None, 
                           src_field=None, trg_field=None,
                           device=None, file_name=None,
                           require_eos=True, args=None):
    model.eval()
    pass

def print_examples(example_iter, model, n=2, max_len=100, 
                   sos_index=1, src_eos_index=None, trg_eos_index=None, 
                   src_field=None, trg_field=None,
                   device=None, file_name=None,
                   require_eos=True, args=None):
    model.eval()
    auto_enc = args["model_type"] == "auto_enc"
    string_out = get_string_rep(example_iter, model,
                                n=n, max_len=max_len,
                                sos_index=sos_index,
                                src_eos_index=src_eos_index,
                                trg_eos_index=trg_eos_index,
                                src_field=src_field,
                                trg_field=trg_field,
                                device=device,
                                require_eos=require_eos,
                                args=args)       
    if auto_enc:
        src_nls, trg_mrs, pred_mrs, pred_nls = string_out
    else:
        src_nls, trg_mrs, pred_mrs = string_out
    string_reps = list(zip(src_nls, trg_mrs, pred_mrs))
    auto_enc = args["model_type"]=="auto_enc"
    if auto_enc:
        string_reps = list(zip(src_nls, trg_mrs, pred_mrs, pred_nls))
    count = 0
    for example_idx, batch in enumerate(string_reps):
        print("Example #%d" % (example_idx+1))
        print("Source NL:", batch[0])
        print("Target MR:", batch[1])
        print("Pred MR:  ", batch[2])
        if auto_enc:
            print("Pred NL:  ", batch[3])
        print()
    
    if file_name is not None:
        with open(file_name, "w") as f:
            for example_idx, batch in enumerate(string_reps):
                f.write(f'Example {example_idx+1}\n')
                f.write(f'Source NL: {batch[0]}\n')
                f.write(f'Target MR: {batch[1]}\n')
                f.write(f'Pred MR: {batch[2]}\n')
                if auto_enc:
                    f.write(f'Pred NL: {batch[3]}\n')

def postfix2infix(s, trg_field, typed=False):
    tokens = s.split(" ")
    stack = []
    #types (top level symbol in the tree; (bool, time, noun))
    if len(s) < 1:
        return []
    for token in tokens:
        if token in trg_field.nouns:
            f_type = (token, "noun")
            stack.insert(0, (token, f_type))
        elif token in trg_field.one_place: #! F G
            op = stack.pop(0)
            op, op_type = op
            if not (token == "G" and op[0] == "F" or\
                    token == "F" and op[0] == "G"):
               op = "( " + op + " )"
            if token in ["F", "G"] or op_type[1] == "time":
                f_type = (token, "time")
            else:
                f_type = (token, "bool")
            stack.insert(0, (token + " " + op, f_type))
        elif token in trg_field.two_place:#& |
            op2 = stack.pop(0)
            op1 = stack.pop(0)
            op1, op1_type = op1
            op2, op2_type = op2
            assert (op1_type[1] != "time" and op2_type[1] != "time") or op2_type[1] == op1_type[1] #same types
            out = " ".join(["(", op1, ")",  token, "(", op2, ")"])
            if token in ["F", "G"] or op1_type[1] == "time" or\
                                      op2_type[1] == "time":
                f_type = (token, "time")
            else:
                f_type = (token, "bool")
            stack.insert(0, (out, f_type))
        else:
            print(s)
            print(token)
            print("unknown token")
            exit()
    if len(stack) == 1:#add parentheses around whole formula
        stack[0] = ("( " + stack[0][0] + " )", stack[0][1])
    
    if typed:
        return stack
    else:
        return [s[0] for s in stack]

def print_memory_usage(msg, device=None):
    print(msg)
    print(f'{torch.cuda.memory_cached(device):,}')
    return torch.cuda.memory_cached(device)

def print_data_info(my_data, src_field, trg_field):
    """ This prints some useful stuff about our data sets. """
    train_data = my_data["train"]
    valid_data = my_data["val"]
    test_data = my_data["test"]

    print("Data set sizes (number of sentence pairs):")
    print('train', len(train_data))
    print('valid', len(valid_data))
    print('test', len(test_data), "\n")

    print("Training Examples:")
    for i in range(5):
        print("src:", train_data[i]['src'])
        print("trg:", train_data[i]['trg'], "\n")

    print("Most common words (src):")
    print("\n".join(["%10s %10d" % x for x in src_field.vocab.freqs.most_common(10)]), "\n")
    print("Most common words (trg):")
    print("\n".join(["%10s %10d" % x for x in trg_field.vocab.freqs.most_common(10)]), "\n")

    print("First 10 words (src):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10])), "\n")
    print("First 10 words (trg):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10])), "\n")

    print("Number of NL words (types):", len(src_field.vocab))
    print("Number of MR words (types):", len(trg_field.vocab), "\n")

def print_gc():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or\
               (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

def print_attn(beam_words, attns, nl_src, src_field, trg_field):
    '''
        beam_words [batch, beam, pred_seq]
        attns [batch, beam, pred_seq, src_seq]
        nl_src [batch, src_seq]
    '''
    batch_i, beam_i = 2, 0
    pred = trg_field.get_string(beam_words[batch_i][beam_i], strip=False).split(" ")
    attn = attns[batch_i][beam_i]
    src_seq = src_field.get_string(nl_src[batch_i], strip=False).split(" ")
    print(pred)
    print(src_seq)
    print(attn)
    for i in range(len(pred)):
        k = min(attn[i].size(0), 3)
        _, max_i = attn[i].topk(k)
        print(pred[i], "->", [(src_seq[max_i[j]], f'{attn[i][max_i[j]].item():.3g}')
                                                  for j in range(k)])

#scratch work
#ltl_args = Namespace(dataset_path='czw_test', formula=formula, is_headless=True, n_sentence=10, num_steps=15, prefix_reward_decay=0.8, recipe_path='/storage/czw/ltl-rl/worlds/craft_recipes_basic.yaml', target_fps=60, update_failed_trans_only=False, use_gui=False)
