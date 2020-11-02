import json
from utils import *

#in_file = "1k_10_env_1_track_no_neg_human"
in_file = "1k_10_env_1_track_no_neg"
#in_data = json.load(open(f'/storage/czw/ltl-rl/ltl/{in_file}.json', "r"))
in_data = json.load(open(f'/storage/czw/rl_parser/{in_file}.json', "r"))
out_file = in_file + "_fruit.json"

data = in_data["data"]

out_data = []
for d in data:
    #d["sentence"] = craft_to_fruit(d["sentence"])
    d["formula"] = craft_to_fruit(d["formula"])
    d["rewritten_formula"] = craft_to_fruit(d["rewritten_formula"])
    out_data.append(d)
    
out_data = {"data": out_data}
json.dump(out_data, open(out_file, "w"))
