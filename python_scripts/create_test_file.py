from utils import *
import pickle
import json
from ltl.worlds.craft_world import Cookbook

path = "1k_10_env_1_track_no_neg"

json_data = json.load(open(f'{path}.json', "r"))

formulas, env_data = [], []
for d in json_data["data"][:100]:
    formulas.append((d["rewritten_formula"], None))
    recipe_path='/storage/czw/rl_parser/ltl/ltl/worlds/craft_recipes_basic.yaml'#NOTE
    cookbook = Cookbook(recipe_path)
    n_features = cookbook.n_kinds+1
    env_d = get_env_data(d, n_features, 0)
    env_data.append(env_d)

pickle.dump(env_data, open(f'ltl/ltl/{path}_env.pkl', 'wb'))
pickle.dump(formulas, open(f'ltl/ltl/{path}_formulas.pkl', 'wb'))
