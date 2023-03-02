from summarizers import MemSum
from tqdm import tqdm
from rouge_score import rouge_scorer
import json
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Test model checkpoint for test data')
parser.add_argument('--test_data_path', help='path to model checkpoint, eg./data/booksum/memsum_data_list_test.jsonl', default="./data/booksum/booksum_memsum_data_list_test.jsonl")
parser.add_argument('--model_path', help='path to model checkpoint, ./model/MemSum_Full/booksum/200dim/run4/model_batch_7301.pt', default="./model/MemSum_Full/booksum/200dim/run4/model_batch_7301.pt")
parser.add_argument('--max_extracted_sentences_per_document', help='The max number of sentences the model can generate as a summary, for booksum, the average length of a gold summary is 55', default="55")
parser.add_argument('--vocabulary_path', help='path to vocabulary eg ./model/glove/vocabulary_200dim.pkl', default="./model/glove/vocabulary_200dim.pkl")
args = parser.parse_args()
rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=True)

memsum_custom_data = MemSum(  args.model_path, args.vocabulary_path, gpu = 0 ,  max_doc_len = 500  )

test_corpus_custom_data = [ json.loads(line) for line in open(args.test_data_path)]

def evaluate( model, corpus, p_stop, max_extracted_sentences, rouge_cal ):
    scores = []
    for data in tqdm(corpus):
        gold_summary = data["summary"]
        extracted_summary = model.extract( [data["text"]], p_stop_thres = p_stop, max_extracted_sentences_per_document = max_extracted_sentences )[0]
        # print('gold summary ->>', gold_summary)
        # print('extracted summary ->>', extracted_summary)
        score = rouge_cal.score( "\n".join( gold_summary ), "\n".join(extracted_summary)  )
        scores.append( [score["rouge1"].fmeasure, score["rouge2"].fmeasure, score["rougeLsum"].fmeasure ] )
    return np.asarray(scores).mean(axis = 0)

print(evaluate( memsum_custom_data, test_corpus_custom_data, 0.6, int(args.max_extracted_sentences_per_document), rouge_cal ))