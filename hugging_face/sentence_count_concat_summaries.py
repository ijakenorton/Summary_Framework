import pathlib
import json





f = open(pathlib.Path(f"./fixed_longt5_data_train.json"),
            encoding='utf-8')
#f = open(pathlib.Path(f"booksum/alignments/chapter-level-summary-alignments/chapter_summary_aligned_train_split.jsonl"), encoding='utf-8')
count = 0
total_text = 0
total_summary = 0
max_length = 0
for line in f: #for each source ->  book
    content = json.loads(line)
    total_text += len(content['text'])
    total_summary += len(content['summary'])
    count += 1
    
print('count ->>> ', count)
print('total_summary ->>> ', total_summary)
print('average summary ->>> ',total_summary/count)
print('total_text ->>> ', total_text)
print('average text ->>> ',total_text/count)
    