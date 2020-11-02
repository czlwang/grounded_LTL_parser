import pickle

in_path = "/storage/czw/rl_parser/jul-27/iml_parse_simple/norm_likelihood_only_cache.pkl"
out_path = "/storage/czw/rl_parser/jul-27/iml_parse_simple/accept_cache.pkl"

in_d = pickle.load(open(in_path, "rb"))

for ex_id in in_d:
    for f in in_d[ex_id]:
        for env_id in in_d[ex_id][f]:
            if in_d[ex_id][f][env_id] > 0:
                in_d[ex_id][f][env_id] = 1

pickle.dump(in_d, open(out_path, "wb"))
