import torch
from transformers import pipeline
import pathlib
import json


summarizer = pipeline(
    "summarization",
    "pszemraj/long-t5-tglobal-base-16384-book-summary",
    device=0 if torch.cuda.is_available() else -1,
)
chap_json = ''
book = ''
f = open(pathlib.Path(f"/data2/coja/hugging_face/booksum/alignments/chapter-level-summary-alignments/old_alignments/chapter_summary_aligned_test_split.jsonl"),
            encoding='utf-8')
for line in f:
    content = json.loads(line)
    if "finished_summaries/sparknotes/The Prince/" in content["summary_path"]:
        path = '../booksum/scripts/' + content["summary_path"]
        chapter = open(pathlib.Path(path),encoding='utf-8')
        text = chapter.read()
        current_json = json.loads(text)
        book += current_json["summary"]
result = summarizer(book)
print(result[0]["summary_text"])
