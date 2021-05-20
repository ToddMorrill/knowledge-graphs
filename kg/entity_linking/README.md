The following steps were taken to download and extract all anchor text statistics with the objective of creating a lookup dictionary to be used in an entity linking system.

```
python download_wikipedia.py \
    --save-dir /Users/tmorrill002/Documents/datasets/wikipedia

python extract_wikipedia.py
```