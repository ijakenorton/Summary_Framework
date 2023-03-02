import torch
import json

train_corpus = [ json.loads(line) for line in open("./data/booksum/fixed_memsum_data_list_train.jsonl") ]

print(len(train_corpus))
print(train_corpus[0].keys())
print(train_corpus[0]["text"][:3])
print(train_corpus[0]["summary"][:3])