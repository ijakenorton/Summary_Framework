from src.data_preprocessing.MemSum.utils import greedy_extract
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Create high rouge scores for input data')
parser.add_argument('--input_path', help='path to training data jsonl, eg./data/booksum/booksum_memsum_data_list_train.jsonl)', default="./data/booksum/booksum_memsum_data_list_train.jsonl")
parser.add_argument('--output_path', help='path to training data jsonl eg ./data/booksum/booksum_memsum_data_list_train_labelled.jsonllll', default="./data/booksum/booksum_memsum_data_list_train_labelled.jsonl")
args = parser.parse_args()
train_corpus = [ json.loads(line) for line in open(args.input_path) ]
for data in tqdm(train_corpus):
    high_rouge_episodes = greedy_extract( data["text"], data["summary"], beamsearch_size = 2 )
    indices_list = []
    score_list  = []

    for indices, score in high_rouge_episodes:
        indices_list.append( indices )
        score_list.append(score)

    data["indices"] = indices_list
    data["score"] = score_list
    # change path to your data
    with open(args.output_path,"w") as f:
        for data in train_corpus:
            f.write(json.dumps(data) + "\n")