import pathlib
import json

from textsum.summarize import Summarizer

summarizer = Summarizer(
    model_name_or_path="pszemraj/long-t5-tglobal-base-16384-book-summary"
)
chap_json = ''
book = ''
f = open(pathlib.Path(f"../booksum/alignments/chapter-level-summary-alignments/fixed_chapter_summaries_test_final.jsonl"),
            encoding='utf-8')
for line in f:
    content = json.loads(line)
    if "finished_summaries/sparknotes/The Prince/" in content["summary_path"]:
        path = '../booksum/scripts/' + content["summary_path"]
        chapter = open(pathlib.Path(path),
            encoding='utf-8')
        text = chapter.read()
        current_json = json.loads(text)
        book += current_json["summary"]
# summarize a long string
out_str = summarizer.summarize_string(book)
print(f"summary: {out_str}")
