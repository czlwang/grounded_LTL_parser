#!/usr/bin/python
import json
from pathlib import Path
import sys

path, out_path = sys.argv[1:]
#path = '/storage/czw/ltl-rl/ltl/sparse_with_distractors_1000_min_path'
#path = '/storage/czw/rl_parser/1k_10_env_1_track_no_neg_fruit_trim.json'
trim = 1
#path = f'/storage/czw/rl_parser/1k_{trim}_env_1_track_no_neg_fruit_trim.json'
#path = f'/storage/czw/rl_parser/1k_{trim}_env_1_track_no_neg_human_fruit_trim.json'
#out_path = '/storage/czw/rl_parser/data/ltl_sparse_no_negation/'
#out_path = f'/storage/czw/rl_parser/data/1k_{trim}_env_1_track_no_neg_human/'
#out_path = f'/storage/czw/rl_parser/data/1k_{trim}_env_1_track_no_neg/'
Path(out_path).mkdir(parents=True, exist_ok=True)
negation = True

train_sentences, train_formulas, train_ltl_data = [], [], []
test_sentences, test_formulas, test_ltl_data = [], [], []
test_split = 150
with open(path) as json_file:
    data = json.load(json_file)
    sentences = [d["sentence"] for d in data["data"]]
    for p in data["data"]:
        formula = p["formula"]
        sentence = p["sentence"]
        if sentences.count(sentence) == 1 and len(test_sentences) < test_split:
            test_sentences.append(sentence)
            test_formulas.append(formula)
            test_ltl_data.append(p)
        else:    
            train_sentences.append(sentence)
            train_formulas.append(formula)
            train_ltl_data.append(p)

if not negation:
    new_train_sentences, new_train_formulas, new_train_ltl_data = [], [], []
    new_test_sentences, new_test_formulas, new_test_ltl_data = [], [], []
    for (sentence, formula, p) in zip(train_sentences, 
                                      train_formulas, train_ltl_data):
        if "!" not in formula:
            new_train_sentences.append(sentence)
            new_train_formulas.append(formula)
            new_train_ltl_data.append(p)
    for (sentence, formula, p) in zip(test_sentences, 
                                      test_formulas, test_ltl_data):
        if "!" not in formula:
            new_test_sentences.append(sentence)
            new_test_formulas.append(formula)
            new_test_ltl_data.append(p)
    train_sentences, train_formulas, train_ltl_data = (new_train_sentences, 
                                                       new_train_formulas, 
                                                       new_train_ltl_data)
    test_sentences, test_formulas, test_ltl_data = (new_test_sentences, 
                                                    new_test_formulas, 
                                                    new_test_ltl_data)

print(f'{len(train_sentences)} train sentences')
print(f'{len(test_sentences)} test sentences')

with open(out_path+"train.src", "w") as f:
    for i in range(len(train_sentences)):
        f.write(train_sentences[i]+"\n")

with open(out_path+"train.trg", "w") as f:
    for i in range(len(train_formulas)):
        f.write(train_formulas[i]+"\n")

train_json_out = {"data": train_ltl_data}
with open(out_path+"train_ltl_data.json", 'w') as json_file:
    json.dump(train_json_out, json_file, indent=4)

with open(out_path+"test.src", "w") as f:
    for i in range(len(test_sentences)):
        f.write(test_sentences[i]+"\n")

with open(out_path+"test.trg", "w") as f:
    for i in range(len(test_formulas)):
        f.write(test_formulas[i]+"\n")

test_json_out = {"data": test_ltl_data}
with open(out_path+"test_ltl_data.json", 'w') as json_file:
    json.dump(test_json_out, json_file, indent=4)
