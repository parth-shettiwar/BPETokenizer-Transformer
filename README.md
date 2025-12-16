# Byte Pair Encoding and Transformer
In this repo, I write a quick optimized implementation of standard BPE algorithm and Transformer architecture. The task was adopted from the stanford course: CS336 assignment
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

# FLOPs and Parameters count
**Input params**

1. vocab_size
2. context_length
3. num_layers
4. d_model
5. num_heads
6. d_ff

Refer this [sheet](https://docs.google.com/spreadsheets/d/1Rl0c0pFwpkKEoTXUP5EMbv3ZgZEMwPsn/edit?usp=sharing&ouid=109510744950242843494&rtpof=true&sd=true)

## Setup

### Environment (copied from assignment)
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.

Run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests

```sh
uv run pytest
```

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

