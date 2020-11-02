#!/usr/bin/env python
import pickle
import bcolz
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical
import random
from utils import *
import time
from rewards import compute_rewards

torch.autograd.set_detect_anomaly(True)
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class BeamNode():
    def __init__(self, prev_h, log_prob, words=[], probs=[]):
        self.words = words
        self.prev_hidden = prev_h
        self.log_prob = log_prob
        self.probs = probs
        self.done = False

    def dummyNode():
        return BeamNode(None, None)

    def eos_reached(self, eos_idx):
        return self.words[-1]==eos_idx

class AutoEncoder(nn.Module):
    def __init__(self, parser, generator):
        super(AutoEncoder, self).__init__()
        self.parser = parser
        self.generator = generator
    
    def forward(self, src_batch, trg_batch, src_lengths, 
                trg_lengths, nl_trg=None,
                nl_trg_lengths=None, teacher_force=False, max_len=None, 
                trg_field=None, src_field=None,require_eos=True,
                ignore_gen=False, args=None):
        '''
        src_batch is [batch_size, seq_len]
        trg_batch is [batch_size, seq_len]
        src_lengths is [batch_size]
        trg_lengths is [batch_size]

        returns
            final_outs [batch_size, beam_width, seq_len, vocab_size]
            final_words [batch_size, beam_width, seq_len]
            attns [batch_size, beam_width, trg_seq_len, src_seq_len]
        '''
        mr_field, nl_field = trg_field, src_field
        encoder_hiddens, encoder_finals = self.parser.encode(src_batch, src_lengths) 
        #print(args["valid_only"])
        parser_out = self.parser.decode(encoder_hiddens, 
                                        encoder_finals,
                                        src_lengths, 
                                        trg_batch, args, max_len=max_len, 
                                        trg_field=trg_field,
                                        teacher_force=teacher_force,
                                        require_eos=require_eos)#TODO
        inter_outs, inter_words, inter_attns = parser_out

        if ignore_gen:
            return inter_outs, inter_words, inter_attns
        #print("here")
        #print(encoder_hiddens.shape)
        #print(encoder_finals[0].shape)

        #teacher forcing!
        gen_beam_width=1#TODO
        gen_args = args.copy()
        gen_args["beam_width"] = gen_beam_width
        gen_args["decode_mode"] = "beam"
        gen_args["epsilon"] = 0
        gen_args["valid_only"] = False
        start = time.time()
        gen_outs = self.generator.decode(encoder_hiddens, 
                                         encoder_finals,
                                         src_lengths, 
                                         nl_trg, gen_args, max_len=None, 
                                         trg_field=nl_field,
                                         teacher_force=teacher_force,
                                         require_eos=False)#TODO

        outs, words, attns = gen_outs

        batch_size, beam_width, out_seq_len = words.shape
        if self.training:#TODO
            words = words.squeeze(1).view(batch_size, beam_width, out_seq_len)
            #[batch*beam, 1, seq_len] -> [batch, beam, out_seq_len]
            outs = outs.squeeze(1).view(batch_size, beam_width, out_seq_len, -1)
            #[batch*beam, 1, seq_len, vocab_size] -> [batch, beam, out_seq_len, vocab]
        else:#TODO
            words = words.view(batch_size, beam_width, gen_beam_width, -1)[:,:,0]
            outs = outs.view(batch_size, beam_width, gen_beam_width, out_seq_len, -1)[:,:,0]
        #print(nl_trg.shape)
        return inter_outs, inter_words, inter_attns, outs, words, attns

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, src, src_lengths):
        return self.encoder(src, src_lengths)

    def expand_beam(self, node, prev_y, trg_field,
                    encoder_hidden, encoder_final, src_length,
                    valid_only=None, beam_width=0):
        '''
            prev_y = [1, 1] tensor
            returns all valid continuations for node
        '''
        hidden = node.prev_hidden
        prev_word = node.words[-1]
        prev_y.fill_(prev_word)
        prev_word_tensor = torch.Tensor(node.words).unsqueeze(0).long()
        valid_idxs = None
        if valid_only:
            valid_idxs = trg_field.get_valid_idx(prev_word_tensor)
        out, hidden, _, = self.decoder(encoder_hidden, encoder_final,
                                         src_length, prev_y, hidden=hidden,
                                         max_len=1,
                                         trg_field=trg_field,
                                         valid_only=valid_only,
                                         valid_idxs=valid_idxs)
        out = out.detach().squeeze()#[vocab_size]
        hidden = (hidden[0].detach().clone(), hidden[1].detach().clone())
        vocab_size = len(out)
        topk_probs, topk_words = out.topk(vocab_size)
        expand_beam = []

        assert not valid_only or len(valid_idxs) == 1

        if valid_only:
            valid_idxs = valid_idxs[0]
        for i in range(len(topk_probs)):
            new_node = BeamNode(hidden, 
                                node.log_prob + topk_probs[i].item(),
                                node.words + [topk_words[i].item()])

            if valid_only and topk_words[i] in valid_idxs:
                expand_beam.append(new_node)
            elif not valid_only:
                expand_beam.append(new_node)
        return expand_beam

    def epsilon_choice(self, beam_nodes, epsilon, 
                       trg_field, beam_width):
        res = []
        vocab_size = len(trg_field.vocab)
        sample_idxs = list(range(len(beam_nodes)))
        beam_width = min(beam_width, len(beam_nodes))
        for k in range(beam_width):
            epsilon_greedy = random.random() < epsilon
            sample_idx = k
            if epsilon_greedy or not k in sample_idxs:
                sample_idx = random.choice(sample_idxs)
            sample_idxs.remove(sample_idx)
            res.append(beam_nodes[sample_idx]) 
        return res

    def prune(self, beam_nodes, trg_field, valid_only=True):
        unfinished_nodes, finished_nodes = [], []
        for node in beam_nodes:
            if node.eos_reached(trg_field.vocab.eos_idx):
                #if trg_field.check_path(node.words) or not valid_only:#TODO
                node.done = True
                finished_nodes.append(node)
            else:
                unfinished_nodes.append(node)
        return unfinished_nodes, finished_nodes

    def pad_beam(self, nodes, beam_width, trg_field, max_len,
                 require_eos, beam_nodes):
        diff = beam_width - len(nodes)
        if require_eos:
            for i in range(diff):
                nodes.append(BeamNode(None, None, words=[]))
        else:
            nodes.extend(beam_nodes[:diff])

        diff = beam_width - len(nodes)#in the edge case when not require_eos, but
                                      #still not enough nodes
        for i in range(diff):
            nodes.append(BeamNode(None, None, words=[]))

        for i in range(len(nodes)):
            pad_len = max_len + 1 - len(nodes[i].words)
            nodes[i].words += [trg_field.vocab.pad_idx]*pad_len
        return nodes

    def beam_decode(self, encoder_hidden, encoder_final, src_length, trg,
                    max_len=None, teacher_force=False, 
                    trg_field=None, require_eos=True, args=None):
        '''
        input
            encoder_hiddens [1, seq_len, num_directions*hidden_size]
            encoder_final [num_layers, 1, num_directions*hidden_size]
            src_length [1]
            trg [1, seq_len]
        output
            beam_words [beam_width, seq_len+1]
        '''
        valid_only = args["valid_only"]
        beam_width = args["beam_width"]
        epsilon = args["epsilon"]

        sos_index = trg_field.vocab.sos_idx 
        start_node = BeamNode(None, 0, [sos_index], [])
        beam_nodes = [start_node]

        if max_len is None:
            max_len = trg.size(1)

        input_device = encoder_hidden.device
        prev_y = torch.ones(1,1).to(input_device).type(torch.long)

        finished_nodes = []
        for i in range(max_len):
            new_nodes = []
            for node in beam_nodes:
                expand_beam = self.expand_beam(node, prev_y, trg_field,
                                               encoder_hidden, encoder_final, 
                                               src_length,
                                               valid_only=valid_only, 
                                               beam_width=beam_width)
                new_nodes.extend(expand_beam)
            beam_nodes = list(sorted(new_nodes, 
                                     key=lambda x: -x.log_prob/len(x.words)))
            beam_nodes = self.epsilon_choice(beam_nodes, epsilon,
                                             trg_field, beam_width)
            beam_nodes = beam_nodes[:beam_width]#unnecessary?
            unfinished_nodes, new_finished_nodes = self.prune(beam_nodes, trg_field,
                                                              valid_only=valid_only)
            beam_nodes = unfinished_nodes
            finished_nodes.extend(new_finished_nodes)
        finished_nodes = list(sorted(finished_nodes, 
                                     key=lambda x: -x.log_prob/len(x.words)))
        finished_nodes = finished_nodes[:beam_width]
        finished_nodes = self.pad_beam(finished_nodes, beam_width,
                                       trg_field, max_len,
                                       require_eos, beam_nodes)
        n = len(finished_nodes)
        cat_words = [torch.tensor(finished_nodes[i].words) for i in range(n)]
        words_tensor = torch.stack(cat_words).squeeze(-1)
        return words_tensor
    
    def sample_decode(self, encoder_hidden, encoder_final, src_length, trg,
                    max_len=None, teacher_force=False, 
                    trg_field=None, args=None):
        '''
        input
            encoder_hiddens [1, seq_len, num_directions*hidden_size]
            encoder_final [num_layers, 1, num_directions*hidden_size]
            src_length [1]
            trg [1, seq_len]
        output
            words_tensor [beam_width, seq_len] 
        '''
        valid_only = args["valid_only"]
        num_samples = args["beam_width"]
        epsilon = args["epsilon"]

        sos_index = trg_field.vocab.sos_idx 
        input_device = encoder_hidden.device

        prev_y = torch.ones(num_samples, 1).fill_(sos_index).long().to(input_device)

        encoder_hidden_repeat = encoder_hidden.repeat(num_samples, 1, 1)
        encoder_final_repeat = (encoder_final[0].repeat(1, num_samples, 1),
                                encoder_final[1].repeat(1, num_samples, 1))
        src_length_repeat = src_length.repeat(num_samples)

        hidden = None

        if max_len is None:
            max_len = trg.size(1)

        words_tensor = torch.ones(num_samples, max_len+1)
        words_tensor[:,0] = sos_index
        for i in range(max_len):
            prev_word_tensor = torch.Tensor(words_tensor[:,:i+1]).long()#NOTE not used
            valid_idxs = None
            #for row in prev_word_tensor:
            #    print(trg_field.get_string(row.long()))#TODO
            if valid_only:
                valid_idxs = trg_field.get_valid_idx(prev_word_tensor)
            out, hidden, _ = self.decoder(encoder_hidden_repeat,
                                          encoder_final_repeat,
                                          src_length_repeat, prev_y, 
                                          hidden=hidden,
                                          max_len=1,
                                          valid_only=valid_only,
                                          trg_field=trg_field,
                                          valid_idxs=valid_idxs)
            out = out.detach()#[beam, 1, vocab]
            out = out.squeeze(1)

            sample = Categorical(torch.exp(out)).sample()#[beam]
            ###epsilon_greedy
            vocab_size = out.shape[-1]
            for k in range(num_samples):
                epsilon_greedy = random.random() < epsilon
                if epsilon_greedy:
                    if valid_only:
                        prev_words = words_tensor[k,:i+1].unsqueeze(0).long()
                        valid_set = valid_idxs[k]
                        random_word = random.choice(valid_set)
                        sample[k] = random_word
                    else:
                        sample[k] = random.randint(0, vocab_size-1)
            ####
                words_tensor[:,i+1] = sample
            #TODO
            #for i in range(num_samples):
            #    s = sample[i]
            #    print(trg_field.get_string(s.unsqueeze(0)))
            #    print(torch.exp(out)[i][s])
            prev_y = sample.detach().unsqueeze(-1)
        input_device = encoder_hidden.device
        words_tensor = words_tensor.to(input_device)
        return words_tensor.long()

    def greedy_decode(self, encoder_hidden, encoder_final, src_length, trg,
                      max_len=None, teacher_force=False, 
                      trg_field=None, args=None):
        '''
        input
            encoder_hiddens [batch, seq_len, num_directions*hidden_size]
            encoder_final [num_layers, batch, num_directions*hidden_size]
            src_length [batch]
            trg [batch, seq_len]
        output
            words_tensor [batch, 1, seq_len] 
        '''
        batch_n, seq_n = trg.shape[0], max_len

        if max_len is None:
            seq_n = trg.shape[1]

        valid_only = args["valid_only"]
        assert args["epsilon"] == 0

        sos_index = trg_field.vocab.sos_idx 
        input_device = encoder_hidden.device

        prev_y = torch.ones(batch_n, 1).fill_(sos_index).long().to(input_device)

        hidden = None

        if max_len is None:
            max_len = trg.size(1)

        words_tensor = torch.ones(batch_n, max_len+1)
        words_tensor[:,0] = sos_index
        for i in range(max_len):
            prev_word_tensor = torch.Tensor(words_tensor[:,:i+1]).long()#NOTE not used
            valid_idxs = None
            if valid_only:
                valid_idxs = trg_field.get_valid_idx(prev_word_tensor)
            out, hidden, _ = self.decoder(encoder_hidden,
                                          encoder_final,
                                          src_length, prev_y, 
                                          hidden=hidden,
                                          max_len=1,
                                          valid_only=valid_only,
                                          trg_field=trg_field,
                                          valid_idxs=valid_idxs)
            out = out.detach()#[batch, 1, vocab]
            out = out.squeeze(1)
            _, greedy_sample = torch.topk(out, 1)
            greedy_sample = greedy_sample.squeeze(-1)
            words_tensor[:,i+1] = greedy_sample
            prev_y = greedy_sample.detach().unsqueeze(-1)
        input_device = encoder_hidden.device
        words_tensor = words_tensor.to(input_device)
        words_tensor = words_tensor.unsqueeze(1)
        return words_tensor.long()

    def batch_decode(self, encoder_hiddens, encoder_finals, src_lengths,
                      trg_batch, max_len=None, teacher_force=False, 
                      trg_field=None, require_eos=True, args=None):
        '''
        input
            encoder_hiddens [batch_size, seq_len, num_directions*hidden_size]
            encoder_finals [num_layers, batch_size, num_directions*hidden_size]
            src_lengths [batch_size]
            trg_batch [batch_size, seq_len]
        output
            [batch_size, beam_width, seq_len] 
        '''
        batch_size, seq_len = encoder_hiddens.shape[0], encoder_hiddens.shape[1]
        final_words = []
        decode_mode=args["decode_mode"]

        assert not (decode_mode=="greedy" and args["beam_width"]!=1)

        if decode_mode=="greedy" and not teacher_force and\
           args["beam_width"]==1:
            beam_words = self.greedy_decode(encoder_hiddens, encoder_finals, 
                                            src_lengths,
                                            trg_batch, max_len=max_len,
                                            teacher_force=teacher_force,
                                            trg_field=trg_field,
                                            args=args)
            return beam_words
        for i in range(batch_size):
           encoder_hidden = encoder_hiddens[i].unsqueeze(0)
           encoder_final = (encoder_finals[0][:,i].unsqueeze(1), 
                            encoder_finals[1][:,i].unsqueeze(1))
           #print("batch_decode")
           #print(trg_batch)
           src_length = src_lengths[i].unsqueeze(0)
           trg = trg_batch[i].unsqueeze(0)
           start = time.time()
           if decode_mode=="sample":
               beam_words = self.sample_decode(encoder_hidden, encoder_final, 
                                               src_length,
                                               trg, max_len=max_len,
                                               teacher_force=teacher_force,
                                               trg_field=trg_field,
                                               args=args)
           if decode_mode=="beam":
               beam_words = self.beam_decode(encoder_hidden, encoder_final,
                                              src_length,
                                              trg, max_len=max_len,
                                              teacher_force=teacher_force,
                                              trg_field=trg_field,
                                              require_eos=require_eos,
                                              args=args)
           print_timing("sample/beam_decode", (time.time() - start), 0)
           final_words.append(beam_words)#[beam_width, seq_len]
        input_device = beam_words.device
        final_words = torch.stack(final_words).type_as(beam_words)
        final_words = final_words.to(input_device)
        return final_words

    def decode(self, encoder_hiddens, encoder_finals, src_lengths, trg_batch,
               args, max_len=None, trg_field=None,
               teacher_force=False, require_eos=True):
        if teacher_force and args["beam_width"]==1:
            out, _, attns = self.decoder(encoder_hiddens, 
                                         encoder_finals,
                                         src_lengths, 
                                         trg_batch, max_len=None, 
                                         trg_field=trg_field,
                                         valid_only=False)
            return out.unsqueeze(1), trg_batch.unsqueeze(1), attns.unsqueeze(1)

        with torch.no_grad():
            start = time.time()
            #print("decode")
            #print(trg_batch)
            final_words = self.batch_decode(encoder_hiddens, 
                                            encoder_finals, 
                                            src_lengths, trg_batch,
                                            teacher_force=teacher_force,
                                            trg_field=trg_field,
                                            require_eos=require_eos,
                                            args=args, max_len=max_len)
            print_timing("batch_decode", (time.time() - start), 0)
        in_pred_tokens = final_words[:,:,:-1]#[batch, beam, seq] 
        out_pred_tokens = final_words[:,:,1:]

        batch_size, beam_size, pred_seq_len = in_pred_tokens.shape
        in_pred_tokens = in_pred_tokens.view(batch_size*beam_size, pred_seq_len)
        input_device = encoder_hiddens.device
        in_pred_tokens = in_pred_tokens.to(input_device)
        #2D matrix. Rows are grouped by beam.

        batch_size, src_seq_len, hidden_size = encoder_hiddens.shape
        encoder_hiddens_repeat = encoder_hiddens.unsqueeze(1)
        encoder_hiddens_repeat = encoder_hiddens_repeat.repeat(1, beam_size, 1, 1)
        encoder_hiddens_repeat = encoder_hiddens_repeat.view(batch_size*beam_size, 
                                                            src_seq_len, hidden_size)
        #[batch_size, seq_len, num_directions*hidden_size] -> 
        # -> [batch_size*beam_size, seq_len, num_directions*hidden_size]
        #so that each sequence is repeated n=beam_size consecutive times

        num_layers, _, _ = encoder_finals[0].shape
        final_cell_repeat = encoder_finals[0].unsqueeze(2)
        final_cell_repeat = final_cell_repeat.repeat(1, 1, beam_size, 1)
        final_cell_repeat = final_cell_repeat.view(num_layers, 
                                                   batch_size*beam_size, 
                                                   hidden_size)
        #[num_layers, batch_size, num_directions*hidden_size]
        # -> [num_layers, batch_size*beam_size, num_directions*hidden_size]

        final_h_repeat = encoder_finals[1].unsqueeze(2)
        final_h_repeat = final_h_repeat.repeat(1, 1, beam_size, 1)
        final_h_repeat = final_h_repeat.view(num_layers, 
                                             batch_size*beam_size, 
                                             hidden_size)

        encoder_finals_repeat = (final_cell_repeat, final_h_repeat)
        src_lengths_repeat = src_lengths.view(-1,1).repeat(1, beam_size).view(-1)
        #[batch_size] -> [batch_size*beam_size]
        #eg [1, 2, 5, 8] -> [1, 1, 2, 2, 5, 5, 8, 8] 

        #[batch_size, seq_len] -> [batch_size*beam_size, seq_len]
        trg_seq_len = trg_batch.shape[-1]
        repeated_trg = trg_batch.unsqueeze(1)
        repeated_trg = repeated_trg.repeat(1, beam_size, 1)
        repeated_trg = repeated_trg.view(batch_size*beam_size, trg_seq_len)

        trg = repeated_trg if teacher_force else in_pred_tokens
        out, _, attns = self.decoder(encoder_hiddens_repeat, 
                                     encoder_finals_repeat,
                                     src_lengths_repeat, 
                                     trg, max_len=None, 
                                     trg_field=trg_field,
                                     valid_only=False)
                                     #NOTE: doesn't seem to train well when 
                                     #valid_only = True. This is because of
                                     #the way I do masking.

        assert not (out != out).any()
        in_seq_len = trg.shape[-1]
        out = out.view(batch_size, beam_size, in_seq_len, -1)

        out_pred_tokens = out_pred_tokens.detach().clone()
        out_pred_tokens = out_pred_tokens.to(input_device)

        attns = attns.view(batch_size, beam_size, in_seq_len, src_seq_len)
        return out, out_pred_tokens, attns


    def forward(self, src_batch, trg_batch, src_lengths, 
                trg_lengths, nl_trg=None, nl_trg_lengths=None,
                teacher_force=False, max_len=None, 
                trg_field=None, src_field=None, require_eos=True,
                ignore_gen=True, args=None):
        '''
        src_batch is [batch_size, seq_len]
        trg_batch is [batch_size, seq_len]
        src_lengths is [batch_size]
        trg_lengths is [batch_size]

        returns
            final_outs [batch_size, beam_width, seq_len, vocab_size]
            final_words [batch_size, beam_width, seq_len]
            attns [batch_size, beam_width, trg_seq_len, src_seq_len]
        '''
        assert not (teacher_force and args["beam_width"] != 1)
        encoder_hiddens, encoder_finals = self.encode(src_batch, src_lengths) 

        out, pred_words, attns = self.decode(encoder_hiddens, 
                                             encoder_finals,
                                             src_lengths, 
                                             trg_batch, args, 
                                             max_len=max_len, 
                                             trg_field=trg_field,
                                             teacher_force=teacher_force,
                                             require_eos=require_eos)#TODO
        return out, pred_words, attns


class Seq2SeqBaseline(EncoderDecoder):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.env_linear1 = nn.Linear(367+13*5, 200)#NOTE hardcode
        self.env_linear2 = nn.Linear(200, 200)#NOTE hardcode

    def decode(self, encoder_hiddens, encoder_finals, src_lengths, trg_batch,
               ltl_data, args, max_len=None, trg_field=None,
               teacher_force=False, require_eos=True):
        #print("here in the decode")
        max_len = 20 #NOTE hardcode

        hidden = None
        batch_size = encoder_hiddens.shape[0]

        dummy_formula = "( G ( gold ) )"
        dummy_formula = replace_symbols(dummy_formula, env_name="Craft")
        gt_actions = ltl_data[0]["actions"][0]["steps"][0]
        ltl_args = get_args() #from utils
        ltl_args.formula = dummy_formula

        cookbook = Cookbook(ltl_args.recipe_path)
        n_features = cookbook.n_kinds+1

        envs = []
        for j in range(batch_size):
            env_data = get_env_data(ltl_data[j], n_features, 0)
            env = sample_craft_env_each(ltl_args,7,7,env_data,None)#NOTE height, width
            envs.append(env)

        observations = []
        for j in range(batch_size):
            env = envs[j]
            obs = env.reset()
            observations.append(obs)

        words_tensors = []
        out_tensors = []

        for i in range(max_len):
            env_features = []
            for j in range(batch_size):
                obs = observations[j]
                features = np.concatenate((obs[0].flatten(), obs[1].flatten()), 0)
                features_tensor = torch.Tensor(features).float()
                env_features.append(features_tensor)
            prev_y = torch.stack(env_features).to(encoder_hiddens.device)
            #prev_y = torch.ones(prev_y.shape).to(encoder_hiddens.device)#TODO
            m = nn.Tanh()
            prev_y = m(self.env_linear1(prev_y))
            prev_y = m(self.env_linear2(prev_y))
            prev_y = prev_y.unsqueeze(1)
            out, hidden, _ = self.decoder(encoder_hiddens,
                                          encoder_finals,
                                          src_lengths, prev_y, 
                                          hidden=hidden,
                                          max_len=1,
                                          valid_only=False,
                                          trg_field=trg_field)
            out = out.squeeze(1)#out [batch, vocab]
            out_tensors.append(out)

            _, greedy_sample = torch.topk(out, 1)
            greedy_sample = greedy_sample.squeeze(-1)
            words_tensors.append(greedy_sample)

            observations = []
            for j in range(batch_size):
                if teacher_force:
                    action = trg_batch[j,i].item()
                else:
                    action = greedy_sample[j].item()
                #print(action)
                env = envs[j]
                #import pdb; pdb.set_trace()
                obs, reward, done, infos = env.step(action)
                observations.append(obs)

        word_tensor = torch.stack(words_tensors, 1)
        word_tensor = word_tensor.detach()
        out_tensor = torch.stack(out_tensors, 1)
        return out_tensor, word_tensor
   
    def forward(self, src_batch, trg_batch, src_lengths, 
                trg_lengths, ltl_data, nl_trg=None, nl_trg_lengths=None,
                teacher_force=False, max_len=None, 
                trg_field=None, src_field=None, require_eos=True,
                ignore_gen=True, args=None):
        '''
        src_batch is [batch_size, seq_len]
        trg_batch is [batch_size, seq_len]
        src_lengths is [batch_size]
        trg_lengths is [batch_size]

        returns
            final_outs [batch_size, beam_width, seq_len, vocab_size]
            final_words [batch_size, beam_width, seq_len]
            attns [batch_size, beam_width, trg_seq_len, src_seq_len]
        '''

        mr_field, nl_field = trg_field, src_field
        encoder_hiddens, encoder_finals = self.encode(src_batch, src_lengths) 
        #print(encoder_hiddens.shape)
        #print(encoder_finals[0].shape)
        out, pred_words = self.decode(encoder_hiddens, 
                                      encoder_finals,
                                      src_lengths, 
                                      trg_batch, 
                                      ltl_data,
                                      args, max_len=max_len, 
                                      trg_field=trg_field,
                                      teacher_force=teacher_force,
                                      require_eos=False)
        return out, pred_words

class Softmax(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Softmax, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x, mask=None):
        """
            In most cases, x is (batch_len x seq_len x hidden_size)
            mask [batch, seq, vocab]
        """
        proj = self.proj(x)
        if mask is not None:
            proj = proj.masked_fill_(mask, -100) #NOTE maybe -inf?
        return F.log_softmax(proj, dim=-1)

class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self, input_size, hidden_size, src_embed,
                 num_layers=1, dropout=0., device=None):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.src_embed = src_embed
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        
    def forward(self, x, lengths):
        """
        input:
            x [batch, seq_len]
        returns
            output [batch_size, seq_len, num_directions*hidden_size]
            encoder_final is a tuple of [num_layers, batch_size, num_directions*hidden_size]

        """
        x = self.src_embed(x)
        total_length = x.size(1)
        packed = pack_padded_sequence(x, lengths, batch_first=True, 
                                      enforce_sorted=False)
        self.rnn.flatten_parameters()
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True,
                                        total_length=total_length)
        # output is of size (batch_length, seq_length, num_directions*hidden_size)

        # we need to manually concatenate the final states for both directions
        # final is a tuple of (hidden, cell)
        # final[0] is the final hidden state. It has size (num_layers*num_directions, batch_length, hidden_size)
        # final[0][0] is the fwd direction for first layer. final[0][2] is forward for second layer and so on.
        fwd_final_hidden = final[0][0:final[0].size(0):2]# [num_layers, batch_len, dim]
        bwd_final_hidden = final[0][1:final[0].size(0):2]
        final_hidden = torch.cat([fwd_final_hidden, bwd_final_hidden], dim=2)  # [num_layers, batch, num_directions*dim]

        fwd_final_cell = final[1][0:final[1].size(0):2]
        bwd_final_cell = final[1][1:final[1].size(0):2]
        final_cell = torch.cat([fwd_final_cell, bwd_final_cell], dim=2)  # [num_layers, batch, num_directions*dim]
        return output, (final_hidden, final_cell)


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""
    
    def __init__(self, emb_size, hidden_size, softmax, trg_embed,
                 attention=None, num_layers=1, dropout=0.0,
                 bridge=True, device=None):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.softmax = softmax
        self.trg_embed = trg_embed

        #the LSTM takes                
        self.rnn = nn.LSTM(emb_size + 2*hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
                 
        # to initialize from the final encoder state
        self.bridge_hidden = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None
        self.bridge_cell = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)

        #input is prev embedding (emb_size), output (hidden_size), and context (num_directions*hidden_size)
        #output is a vector of size (hidden_size). This vector is fed through a softmax layer to get a 
        #distribution over vocab
        self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size,
                                          hidden_size, bias=False)
        
    def mask_invalid(self, batch_size, valid_idxs, vocab_size):
        '''
            valid_idxs [batch, *] 
                list of (list of valid continuation indices) for each word

            out:
                masked_out [beam_vocab] where only valid_idxs are False
        '''
        mask = []
        for i in range(batch_size):
            mask_row = [True]*vocab_size
            for j in range(len(valid_idxs[i])):
                mask_row[valid_idxs[i][j]] = False 
            mask.append(torch.tensor(mask_row))
        mask = torch.stack(mask, dim=0)
        return mask

    def forward_step(self, prev_embed, encoder_hidden, 
                     src_lengths, proj_key, hidden):
        """Perform a single decoder step (1 word)
           input
               prev_embed [batch_size, 1,  emb_size] is the target previous word
           encoder_hidden [batch_size, enc_seq_len, num_directions*hidden_size] 
                It is the output of the encoder
           proj_key [batch_size, enc_seq_len, proj_dim=hidden_size]
           hidden[0] [num_layers, batch_size,  hidden_size]
        returns
            output [batch, seq_len=1, hidden_size]
            hidden[0] [num_layers, batch_size, hidden_size]
            pre_output [batch_size, seq_len=1, hidden_size]
        """
        # compute context vector using attention mechanism
        #we only want the hidden, not the cell state of the lstm CZW, hence the hidden[0]
        query = hidden[0][-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, src_lengths=src_lengths)

        # update rnn hidden state
        # context is [batch, 1, num_directions*hidden_size]
        # the lstm takes the previous target embedding and
        # the attention context as input
        rnn_input = torch.cat([prev_embed, context], dim=2)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(rnn_input, hidden)
        
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        # pre_output is actually used to compute the prediction
        return output, hidden, pre_output
    
    def forward(self, encoder_hidden, encoder_final, src_lengths, trg,
                hidden=None, max_len=None, valid_only=False, trg_field=None,
                valid_idxs=None):
        """Unroll the decoder one step at a time.
        because we're doing beam decoding: assume batch_size=1
        Input
            encoder_hidden is [batch_len, src_max_seq_len, 
                               num_directions*hidden_layer_size]
            encoder_final is a tuple of final hidden and final cell state. 
              each state is [num_layers, batch, num_directions*hidden_layer_size]
            src_lengths is [batch_len]
            trg is [batch_size, seq_len]
            hidden is the hidden state of the previous step
        Output
            probs [batch, seq_len, vocab_size]
            hidden is tuple of [num_layers, batch_size, hidden_size]
            attns [batch_size*beam_size, pred_seq_len, src_seq_len]
        """
        batch_size, trg_seq_len = trg.shape[0], trg.shape[1]
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg.size(1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)

        probs = []
        attns = []
        pre_outputs = []
        trg_embed = self.trg_embed(trg)
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(prev_embed,
                                                           encoder_hidden, 
                                                           src_lengths,
                                                           proj_key,
                                                           hidden)

            #output [batch, 1, hidden_size]
            valid_mask=None
            if valid_only:
                #if you have a trg to teacher force of, use that
                if valid_idxs is not None:
                    valid_indices = valid_idxs
                else:
                    valid_indices = trg_field.get_valid_idx(trg[:,:i+1])
                vocab_size = len(trg_field.vocab)
                batch_size = output.shape[0]
                valid_mask = self.mask_invalid(batch_size, 
                                               valid_indices, vocab_size)
                input_device = pre_output.device
                valid_mask = valid_mask.to(input_device)
                valid_mask = valid_mask.unsqueeze(1) #[batch, 1, vocab_size]
            probs.append(self.softmax(pre_output, valid_mask))
            attns.append(self.attention.alphas.clone().detach())
            pre_outputs.append(pre_output)
        attns = torch.cat(attns, 1)
        probs = torch.cat(probs, 1)
        pre_outputs = torch.cat(pre_outputs, 1)
        return probs, hidden, attns.clone().detach()

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return (torch.tanh(self.bridge_hidden(encoder_final[0])),
                torch.tanh(self.bridge_cell(encoder_final[1])))

class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) ttention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, src_lengths=None):
        '''
            Input:
               query [batch, 1, hidden_size]
               proj_key [batch, enc_seq_len, hidden_size]
               value [batch, enc_seq_len, num_layers*hidden_size]
            Output:
               context [batch, 1, num_layers*hidden_size]
               alphas [batch, 1, enc_seq_len]
        '''
        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        mask = []
        max_len = scores.size(2)
        for length in src_lengths:
            mask.append([[1]*length.item() + [0]*(max_len - length.item())])

        input_device = scores.device
        tensor_mask = torch.Tensor(mask).ne(1)
        tensor_mask = tensor_mask.to(input_device)
     
        scores.data.masked_fill_(tensor_mask, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

def make_weights_matrix(field, glove):
    emb_dim = 200 #TODO hardcode
    matrix_len = len(field.vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0

    for i, word in enumerate(field.vocab.itos):
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))

    return weights_matrix

def make_autoencoder(emb_size=256, hidden_size=512, num_layers=1, dropout=0.0,
                     nl_field=None, mr_field=None):
    nl_weights, mr_weights = make_weights(nl_field, mr_field)

    nl_vocab_size, mr_vocab_size = len(nl_field.vocab), len(mr_field.vocab)
    parse_nl_embed = nn.Embedding(nl_vocab_size, emb_size)
    parse_nl_embed.weight.data.copy_(torch.from_numpy(nl_weights))
    parse_nl_embed.weight.requires_grad = False
    parse_mr_embed = nn.Embedding(mr_vocab_size, emb_size)
    parse_mr_embed.weight.data.copy_(torch.from_numpy(mr_weights))

    gen_nl_embed = nn.Embedding(nl_vocab_size, emb_size)
    gen_nl_embed.weight.data.copy_(torch.from_numpy(nl_weights))
    gen_mr_embed = nn.Embedding(mr_vocab_size, emb_size)

    attention_1 = BahdanauAttention(hidden_size)
    attention_2 = BahdanauAttention(hidden_size)
    parser =    EncoderDecoder(Encoder(emb_size, hidden_size, parse_nl_embed,
                                       num_layers=num_layers, dropout=dropout), 
                               Decoder(emb_size, hidden_size, 
                                       Softmax(hidden_size, mr_vocab_size), 
                                       parse_mr_embed, attention_1, 
                                       num_layers=num_layers, dropout=dropout))
    generator = EncoderDecoder(Encoder(emb_size, hidden_size, gen_mr_embed,
                                       num_layers=num_layers, dropout=dropout), 
                               Decoder(emb_size, hidden_size, 
                                       Softmax(hidden_size, nl_vocab_size), 
                                       gen_nl_embed, attention_2, 
                                       num_layers=num_layers, dropout=dropout))

    autoencoder = AutoEncoder(parser, generator)
    return autoencoder

def make_weights(src_field, trg_field):
    src_vocab_size, trg_vocab_size = len(src_field.vocab), len(trg_field.vocab)
    glove_path = "/storage/czw/rl_parser/glove"
    vectors = bcolz.open(f'{glove_path}/6B.200.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.200_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.200_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}

    src_weights = make_weights_matrix(src_field, glove)
    trg_weights = make_weights_matrix(trg_field, glove)
    return src_weights, trg_weights

def make_seq2seq_base(emb_size=256, hidden_size=512, num_layers=1, dropout=0.0,
                      src_field=None, trg_field=None):

    src_weights, trg_weights = make_weights(src_field, trg_field)

    src_vocab_size, trg_vocab_size = len(src_field.vocab), len(trg_field.vocab)
    src_embed = nn.Embedding(src_vocab_size, emb_size)
    src_embed.weight.data.copy_(torch.from_numpy(src_weights))
    src_embed.weight.requires_grad = False
    trg_embed = nn.Linear(emb_size, emb_size)#TODO hardcode

    attention = BahdanauAttention(hidden_size)

    parser = Seq2SeqBaseline(Encoder(emb_size, hidden_size, src_embed,
                                    num_layers=num_layers, 
                                    dropout=dropout), 
                             Decoder(emb_size, hidden_size, 
                                     Softmax(hidden_size, trg_vocab_size), 
                                     trg_embed, attention, 
                                     num_layers=num_layers, 
                                     dropout=dropout))
    return parser

def make_parser(emb_size=256, hidden_size=512, num_layers=1, dropout=0.0,
                 src_field=None, trg_field=None):

    src_weights, trg_weights = make_weights(src_field, trg_field)

    src_vocab_size, trg_vocab_size = len(src_field.vocab), len(trg_field.vocab)
    src_embed = nn.Embedding(src_vocab_size, emb_size)
    src_embed.weight.data.copy_(torch.from_numpy(src_weights))
    src_embed.weight.requires_grad = False
    trg_embed = nn.Embedding(trg_vocab_size, emb_size)
    trg_embed.weight.data.copy_(torch.from_numpy(trg_weights))

    attention = BahdanauAttention(hidden_size)

    parser = EncoderDecoder(Encoder(emb_size, hidden_size, src_embed,
                                    num_layers=num_layers, dropout=dropout), 
                            Decoder(emb_size, hidden_size, 
                                    Softmax(hidden_size, trg_vocab_size), 
                                    trg_embed, attention, 
                                    num_layers=num_layers, dropout=dropout))
    return parser

