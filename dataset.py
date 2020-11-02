#!/usr/bin/env python
import torch
import json
from torch.utils.data import Dataset
from utils import postfix2infix

class TextCollator(object):
    def __init__(self, pad_idx=1, device=None):
        self.device=device
        self.pad_idx = pad_idx

    def __call__(self, data):
        pad_id = self.pad_idx
        nl_srcs = [d["nl_src"] for d in data]
        ids = [d["id"] for d in data]
        ltl_data = [d["ltl_data"] for d in data]
        mr_srcs = [d["mr_src"] for d in data]
        mr_trgs = [d["mr_trg"] for d in data]
        mr_trg_ys = [d["mr_trg_y"] for d in data]
        nl_trg_ys = [d["nl_trg_y"] for d in data]
        nl_trgs = [d["nl_trg"] for d in data]
        nl_src_lengths = torch.cat([d["nl_src_length"] for d in data])
        nl_trg_lengths = torch.cat([d["nl_trg_length"] for d in data])
        mr_trg_lengths = torch.cat([d["mr_trg_length"] for d in data])
        mr_src_lengths = torch.cat([d["mr_src_length"] for d in data])
        nseqs = sum([d["nseqs"] for d in data])
        ntokens = sum([d["ntokens"] for d in data])

        max_src_len = max(nl_src_lengths).item()
        max_trg_len = max(mr_trg_lengths).item()

        pad_nl = pad_seqs(nl_srcs, max_src_len, pad_id)
        nl_src = torch.tensor(pad_nl).to(self.device).long()

        pad_mr = pad_seqs(mr_trgs, max_trg_len, pad_id)
        mr_trg = torch.tensor(pad_mr).to(self.device).long()

        pad_mr_src = pad_seqs(mr_srcs, max_trg_len, pad_id)
        mr_src = torch.tensor(pad_mr_src).to(self.device).long()
    
        pad_mr_y = pad_seqs(mr_trg_ys, max_trg_len, pad_id)
        mr_trg_y = torch.tensor(pad_mr_y).to(self.device).long()

        pad_nl_y = pad_seqs(nl_trg_ys, max_src_len, pad_id)
        nl_trg_y = torch.tensor(pad_nl_y).to(self.device).long()

        pad_nl_trg = pad_seqs(nl_trgs, max_src_len, pad_id)
        nl_trg = torch.tensor(pad_nl_trg).to(self.device).long()

        return {"nl_src": nl_src,
                "nl_src_lengths": nl_src_lengths,
                "nl_trg": nl_trg,
                "nl_trg_lengths": nl_trg_lengths,
                "mr_trg": mr_trg,
                "mr_src": mr_src,
                "mr_trg_lengths": mr_trg_lengths,
                "mr_src_lengths": mr_src_lengths,
                "mr_trg_y": mr_trg_y,
                "nl_trg_y": nl_trg_y,
                "nseqs": nseqs,
                "ntokens": ntokens,
                "ltl_data": ltl_data,
                "id": ids}

class ActionCollator(object):
    def __init__(self, pad_idx=1, device=None):
        self.device=device
        self.pad_idx = pad_idx

    def __call__(self, data):
        pad_id = self.pad_idx
        nl_srcs = [d["nl_src"] for d in data]
        ids = [d["id"] for d in data]
        ltl_data = [d["ltl_data"] for d in data]
        acts_srcs = [d["acts_src"] for d in data]
        acts_trgs = [d["acts_trg"] for d in data]
        acts_trg_ys = [d["acts_trg_y"] for d in data]
        nl_trg_ys = [d["nl_trg_y"] for d in data]
        nl_trgs = [d["nl_trg"] for d in data]
        nl_src_lengths = torch.cat([d["nl_src_length"] for d in data])
        nl_trg_lengths = torch.cat([d["nl_trg_length"] for d in data])
        acts_trg_lengths = torch.cat([d["acts_trg_length"] for d in data])
        acts_src_lengths = torch.cat([d["acts_src_length"] for d in data])
        nseqs = sum([d["nseqs"] for d in data])
        ntokens = sum([d["ntokens"] for d in data])

        max_src_len = max(nl_src_lengths).item()
        max_trg_len = max(acts_trg_lengths).item()

        pad_nl = pad_seqs(nl_srcs, max_src_len, pad_id)
        nl_src = torch.tensor(pad_nl).to(self.device).long()

        pad_acts = pad_seqs(acts_trgs, max_trg_len, pad_id)
        acts_trg = torch.tensor(pad_acts).to(self.device).long()

        pad_acts_src = pad_seqs(acts_srcs, max_trg_len, pad_id)
        acts_src = torch.tensor(pad_acts_src).to(self.device).long()
    
        pad_acts_y = pad_seqs(acts_trg_ys, max_trg_len, pad_id)
        acts_trg_y = torch.tensor(pad_acts_y).to(self.device).long()

        pad_nl_y = pad_seqs(nl_trg_ys, max_src_len, pad_id)
        nl_trg_y = torch.tensor(pad_nl_y).to(self.device).long()

        pad_nl_trg = pad_seqs(nl_trgs, max_src_len, pad_id)
        nl_trg = torch.tensor(pad_nl_trg).to(self.device).long()

        return {"nl_src": nl_src,
                "nl_src_lengths": nl_src_lengths,
                "nl_trg": nl_trg,
                "nl_trg_lengths": nl_trg_lengths,
                "acts_trg": acts_trg,
                "acts_src": acts_src,
                "acts_trg_lengths": acts_trg_lengths,
                "acts_src_lengths": acts_src_lengths,
                "acts_trg_y": acts_trg_y,
                "nl_trg_y": nl_trg_y,
                "nseqs": nseqs,
                "ntokens": ntokens,
                "ltl_data": ltl_data,
                "id": ids}

def infix2postfix(s, trg_field):
    #assume that s is fully parenthesized

    stack = []
    tokens = s.strip().split(" ")
    out = []
    for token in tokens:
        if token in trg_field.nouns:
            out.append(token)
        elif token in trg_field.one_place or token in trg_field.two_place:
            stack.insert(0, token)    
        elif token == "(":
            stack.insert(0, token)
        elif token == ")":
            while stack[0] != "(":
                out.append(stack.pop(0))
            stack.pop(0)
        else:
            print("unknown token")
    res = ' '.join(out)
    return res

class ParseDataset(Dataset):
    def __init__(self, src_strings, trg_strings, 
                 src_field, trg_field, device=None,
                 ltl_json=None, ltl=False, ids=None):
        self.src_strings = src_strings
        self.trg_strings = trg_strings
        self.src_field = src_field
        self.trg_field = trg_field
        self.device = device
        self.ltl_data = ltl_json
        self.ltl = ltl
        self.ids = ids

    def __len__(self):
        return len(self.src_strings)
    
    def __getitem__(self, idx):
        """
            nl has start and end token
            mr has start and end token
            self.nl_src has no start token but end token
            self.nl_trg has start but no end token
            self.nl_trg_y has end but no start token
            self.mr_trg has start but no end token
            self.mr_trg_y has no start but end token
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        src = self.src_strings[idx]
        trg = self.trg_strings[idx]

        if self.ltl:
            trg = infix2postfix(trg, self.trg_field) 

        src_ids = self.src_field.to_idxs(src)
        trg_ids = self.trg_field.to_idxs(trg)

        src_sos_idx = self.src_field.vocab.stoi[self.src_field.vocab.sos_token]
        src_eos_idx = self.src_field.vocab.stoi[self.src_field.vocab.eos_token]
        nl = [src_sos_idx] + src_ids + [src_eos_idx]
        nl_length = len(nl)
        nl_src = nl[1:]
        nl_src_length = len(nl_src)
        nl_trg = nl[:-1]
        nl_trg_length = len(nl_trg)
        nl_trg_y = nl[1:]

        trg_sos_idx = self.trg_field.vocab.stoi[self.trg_field.vocab.sos_token]
        trg_eos_idx = self.trg_field.vocab.stoi[self.trg_field.vocab.eos_token]
        mr = [trg_sos_idx] + trg_ids + [trg_eos_idx]
        mr_src = mr[1:]
        mr_length = len(mr)
        mr_trg = mr[:-1]
        mr_trg_length = len(mr_trg)
        mr_src_length = len(mr_src)
        mr_trg_y = mr[1:]
        
        ntokens = len(mr_trg_y)
        nseqs = 1

        _id = idx if self.ids is None else self.ids[idx]
        nl_src_length = torch.ones(1).fill_(nl_src_length).to(self.device).long()
        nl_trg_length = torch.ones(1).fill_(nl_trg_length).to(self.device).long()
        mr_trg_length = torch.ones(1).fill_(mr_trg_length).to(self.device).long()
        mr_src_length = torch.ones(1).fill_(mr_src_length).to(self.device).long()
        return {"src": src,
                "trg": trg,
                "nl_src": nl_src,
                "nl_trg": nl_trg,
                "nl_trg_length": nl_trg_length,
                "nl_src_length": nl_src_length,
                "mr_trg": mr_trg,
                "mr_src": mr_src,
                "mr_trg_length": mr_trg_length,
                "mr_src_length": mr_src_length,
                "mr_trg_y": mr_trg_y,
                "nl_trg_y": nl_trg_y,
                "ntokens": ntokens,
                "nseqs": nseqs,
                "ltl_data": self.ltl_data["data"][idx],
                "id": _id}

class ActionDataset(Dataset):
    def __init__(self, src_strings, trg_strings, 
                 src_field, trg_field, device=None,
                 ltl_json=None, ltl=False, ids=None):
        self.src_strings = src_strings
        self.trg_strings = trg_strings
        self.src_field = src_field
        self.trg_field = trg_field
        self.device = device
        self.ltl_data = ltl_json
        self.ids = ids

    def __len__(self):
        return len(self.src_strings)
    
    def __getitem__(self, idx):
        """
            nl is natural language
            acts is actions

            nl has start and end token
            self.nl_src has no start token but end token
            self.nl_trg has start but no end token
            self.nl_trg_y has end but no start token
            self.acts_trg has no start and no end token
            self.acts_trg_y has no start and no end token
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        src = self.src_strings[idx]
        trg = self.trg_strings[idx]

        src_ids = self.src_field.to_idxs(src)
        trg_ids = self.trg_field.to_idxs(trg)

        src_sos_idx = self.src_field.vocab.stoi[self.src_field.vocab.sos_token]
        src_eos_idx = self.src_field.vocab.stoi[self.src_field.vocab.eos_token]
        nl = [src_sos_idx] + src_ids + [src_eos_idx]
        nl_length = len(nl)
        nl_src = nl[1:]
        nl_src_length = len(nl_src)
        nl_trg = nl[:-1]
        nl_trg_length = len(nl_trg)
        nl_trg_y = nl[1:]

        acts = trg_ids
        acts_src = acts
        acts_length = len(acts)
        acts_trg = acts
        acts_trg_length = len(acts_trg)
        acts_src_length = len(acts_src)
        acts_trg_y = acts[1:]
        
        ntokens = len(acts_trg_y)
        nseqs = 1

        _id = idx if self.ids is None else self.ids[idx]
        nl_src_length = torch.ones(1).fill_(nl_src_length).to(self.device).long()
        nl_trg_length = torch.ones(1).fill_(nl_trg_length).to(self.device).long()
        acts_trg_length = torch.ones(1).fill_(acts_trg_length).to(self.device).long()
        acts_src_length = torch.ones(1).fill_(acts_src_length).to(self.device).long()
        return {"src": src,
                "trg": trg,
                "nl_src": nl_src,
                "nl_trg": nl_trg,
                "nl_trg_length": nl_trg_length,
                "nl_src_length": nl_src_length,
                "acts_trg": acts_trg,
                "acts_src": acts_src,
                "acts_trg_length": acts_trg_length,
                "acts_src_length": acts_src_length,
                "acts_trg_y": acts_trg_y,
                "nl_trg_y": nl_trg_y,
                "ntokens": ntokens,
                "nseqs": nseqs,
                "ltl_data": self.ltl_data["data"][idx],
                "id": _id}

def pad_seqs(seqs, max_len, pad_idx):
    pads = []
    for ids in seqs:
       padded = ids + [pad_idx]*(max_len - len(ids))
       pads.append(padded)
    return pads


