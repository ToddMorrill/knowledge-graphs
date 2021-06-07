The following steps were taken to download and extract all anchor text statistics with the objective of creating a lookup dictionary to be used in an entity linking system.

```
python download_wikipedia.py \
    --save-dir /Users/tmorrill002/Documents/datasets/wikipedia

python extract_wikipedia.py \
    --wiki-dir /Users/tmorrill002/Documents/datasets/wikipedia/20210401/ \
    --save-dir /Users/tmorrill002/Documents/datasets/wikipedia/links_20210401 \
    --config extract_wikipedia.yaml

python entity_db.py \
    --save-dir /Users/tmorrill002/Documents/datasets/wikipedia/20210401_sqlite \
    populate \
    --link-dir /Users/tmorrill002/Documents/datasets/wikipedia/links_20210401

python entity_db.py \
    --save-dir /Users/tmorrill002/Documents/datasets/wikipedia/20210401_sqlite \
    query \
    --queries 'New York' 'Mumbai' 'Shanghai'
```