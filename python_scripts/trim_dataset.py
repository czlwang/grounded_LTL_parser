import json
import sys

in_file, out_file = sys.argv[1:]
#in_file = "1k_10_env_1_track_no_neg_fruit"
#in_file = "1k_10_env_1_track_no_neg_human_fruit"
#in_data = json.load(open(f'/storage/czw/ltl-rl/ltl/{in_file}.json', "r"))
in_data = json.load(open(f'{in_file}.json', "r"))

trim = 1
#out_file = f'1k_{trim}_env_1_track_no_neg_human_fruit' + "_trim.json"
#out_file = f'1k_{trim}_env_1_track_no_neg_fruit' + "_trim.json"

out_data = []
data = in_data["data"]
for idx in range(len(data)):
    d = data[idx]
    d["actions"] = d["actions"][:trim]
    d["envs"] = d["envs"][:trim]
    out_data.append(d)
    
out_data = {"data": out_data}
json.dump(out_data, open(out_file, "w"))

