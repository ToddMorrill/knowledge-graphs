# knowledge-graphs
Experimental approaches to constructing knowledge graphs and using them for reasoning tasks

## Pipeline
```
# download CoNLL-2003 tags and prepare train, validation, and test data using CoNLL-2003 scripts
python download_unzip.py \
    --url https://www.clips.uantwerpen.be/conll2003/ner.tgz \
    --save-directory /Users/tmorrill002/Documents/datasets/conll/raw \
    --reuters-file-path /Users/tmorrill002/Documents/datasets/reuters/rcv1.tar.xz

# parse the data into csv files for use with Pandas
python parse.py \
    --data-directory /Users/tmorrill002/Documents/datasets/conll/raw/ner \
    --save-directory /Users/tmorrill002/Documents/datasets/conll/transformed

# run a baseline algorithm to evaluate performance of a naive approach
# this is the approach detailed in the CoNLL-2003 paper
python baseline.py \
    --data-directory /Users/tmorrill002/Documents/datasets/conll/transformed

# run a vanilla LSTM and evaluate the results
python train.py \
    --config configs/baseline.yaml
```

#### Virtual environment setup
Still using Python 3.8 for now until more libaries are compatible with Python 3.9 (as of December 22, 2020).

```
# use Python 3.8 installed with homebrew
virtualenv .venv -p /usr/local/opt/python@3.8/bin/python3

source .venv/bin/activate
pip install -r requirements.txt
```