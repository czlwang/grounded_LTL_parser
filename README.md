# grounded_LTL_parser
This is the code for [Learning a natural-language to LTL executable semantic parser for grounded robotics](https://arxiv.org/abs/2008.03277)

Attention mechanism implementation from [1]

## Requirements
 - Python 3.7
 - [Torch 1.2.0](https://pytorch.org/get-started/locally/)
 - [OpenAI Baselines](https://github.com/openai/baselines)
 - [Spot 2.9.3](https://spot.lrde.epita.fr/install.html)

## Install
The execution environment and other python libraries are needed. We recommend creating a new virtual environment before proceeding.  In the cloned directory, run the following:
 1. `pip install -r requirements.txt`
 2. ~~`git submodule add git@github.com:czlwang/ltl-environment-dev.git ltl`~~
 3. ~~`git submodule init`~~
 4. ~~`git submodule update`~~
 5. ~~`pip install -e ltl`~~


## Data
~~The data is at `victoria.csail.mit.edu:/storage/czw/rl_parser/data`~~

## Run
`./reinforce.py configs/rl_parse_config.yml`(This will run for a long time. If you just want to check if everything is working, you set `train_split` to be smaller in the config.)

## Important config flags
- `data_dir` should be set to wherever you put the data
- `train_mode` is either `iml`, `rl`, or `ml` for Iterative Maximum Likelihood, Reinforcement Learning, or Maximum Likelihood respectively
- `eval_only` is True if running evaluation only. In that case, make sure to set `load_pretrained` and `pretrained` as well.

## Contents

- reinforce.py contains our code for training
	- `iter_ml()` contains the Iterative Maximum Likelihood procedure
- model.py contains our sequence-to-sequence model
- rewards.py contains our rewards methods
	- `compute_ltl_rewards()` contains the reward computation as described in Section 4

## Citations
[1] J. Bastings. 2018. The Annotated Encoder-Decoder with Attention. https://bastings.github.io/annotated_encoder_decoder/
