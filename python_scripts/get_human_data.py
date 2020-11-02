import json
import random
from utils import *

in_file = "1k_10_env_1_track_no_neg"
in_data = json.load(open(f'/storage/czw/ltl-rl/ltl/{in_file}.json', "r"))

human_file = "/storage/czw/ltl_gif2/public/human_out_0718.json"
human_list = json.load(open(human_file, "r"))

human_data = {}
done = []
for h in human_list:
    idx = int(h["example_id"])
    done.append(idx)
    if "dataset" in h and h["dataset"] == in_file and h["sentence"] != "":#NOTE
        human_data[idx] = h["sentence"]

out_file = in_file + "_human.json"

data = in_data["data"]

out_data, out_data_human = [], []
for idx in range(len(data)):
    d = data[idx]
    if idx in human_data:
        old_d = d.copy()
        out_data.append(old_d)
        d["sentence"] = human_data[idx]
        d["formula"] = craft_to_fruit(d["formula"])
        d["rewritten_formula"] = craft_to_fruit(d["rewritten_formula"])
        out_data_human.append(d)
    
src_trg = list(zip(out_data, out_data_human))
random.shuffle(src_trg)
out_data, out_data_human = zip(*src_trg)

out_data = {"data": out_data}
out_data_human = {"data": out_data_human}
json.dump(out_data_human, open(out_file, "w"))
json.dump(out_data, open(in_file + ".json", "w"))
