import spot
import json
from collections import Counter
import statistics

#path = f'/storage/czw/rl_parser/1k_3_env_1_track_no_neg_human_fruit_trim.json'
path = f'/storage/czw/rl_parser/1k_10_env_1_track_no_neg_fruit_trim.json'
a = json.load(open(path, "r"))
data = a["data"]
fs = [d["formula"] for d in data]
sentences = [d["sentence"] for d in data]
f_len = lambda formula: len([token for token in formula.split(' ') if token not in ['(', ')']])
classes = list(map(spot.mp_class, fs))
print(Counter(classes).values())
print(Counter(classes).keys())
lens = list(map(f_len, fs)) 
sen_lens = list(map(f_len, sentences)) 

print("f len", statistics.mean(lens), statistics.stdev(lens))
print("sentence len", statistics.mean(sen_lens), statistics.stdev(sen_lens))
