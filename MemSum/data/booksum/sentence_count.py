import pathlib
import json





f = open(pathlib.Path(f"./memsum_data_list_train.jsonl"),
            encoding='utf-8')
#f = open(pathlib.Path(f"booksum/alignments/chapter-level-summary-alignments/chapter_summary_aligned_train_split.jsonl"), encoding='utf-8')
count = 0
total = 0
for line in f: #for each source ->  book
    content = json.loads(line)
    total += len(content['summary'])

    count += 1
    
print('total ->>> ', total)
print('count ->>> ', count)
print('average ->>> ',total/count)
    