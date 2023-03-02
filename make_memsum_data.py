import sys
import pathlib
import json
from nltk import tokenize
import argparse
def get_books(split):
    """
    Reads from a JSONL file containing book-level summaries, extracts the summary text for each book,
    and stores the summary text in a dictionary using the source bid as the key.
    
    Parameters:
        split (str): The name of the split (e.g. 'train', 'val', 'test')
        
    Returns:
        A dictionary where the keys are the source bid and the values are dictionaries with 'summary' and 'text' keys
        The text is stored in a list of sentencized sentences.
    """
    book_data = {}
     # Change the path below to the path of your data
    book_path = pathlib.Path(f"../../alignments/book-level-summary-alignments/fixed_book_summaries_{split}_final.jsonl")
    
    with open(book_path, encoding='utf-8') as book_file:
        for line in book_file:
            content = json.loads(line)
            source_bid = content['source'] + content['bid']
            path = content['summary_path']
            
            if source_bid not in book_data:
                book_data[source_bid] = {'summary': [], 'text': []}
                
            try:
                with open(pathlib.Path(f"../../scripts/{path}"), encoding='utf-8') as text_file:
                    chap = json.load(text_file)
                    book_data[source_bid]['summary'] = tokenize.sent_tokenize(chap['summary'])
                    
            except (FileNotFoundError, json.JSONDecodeError):
                print('Could not find or parse text at path:', path)
                
    return book_data


def get_texts(book_data, split):
    """
    Reads from a JSONL file containing chapter-level summaries, extracts the summary text for each chapter,
    and appends the summary text to the appropriate book's text in the dictionary created by get_books().
    
    Parameters:
        book_data (dict): A dictionary created by get_books() with the source bid as the key and summary and text
        dictionaries as the value.
        split (str): The name of the split (e.g. 'train', 'val', 'test')
        
    Returns:
        The same dictionary as book_data, but with the 'text' value updated to include the chapter summaries.
    """
    # Change the path below to the path of your data
    text_path = pathlib.Path(f"../../alignments/chapter-level-summary-alignments/fixed_chapter_summaries_{split}_final.jsonl")
    
    with open(text_path, encoding='utf-8') as text_file:
        for line in text_file:
            content = json.loads(line)
            source_bid = content['source'] + content['bid']
            path = content['summary_path']
            
            if source_bid not in book_data:
                continue
                
            try:
                with open(pathlib.Path(f"../../scripts/{path}"), encoding='utf-8') as chap_file:
                    chap = json.load(chap_file)
                    book_data[source_bid]['text'].append(tokenize.sent_tokenize(chap['summary']))
                    
            except (FileNotFoundError, json.JSONDecodeError):
                print('Could not find or parse text at path:', path)
                
    return book_data


def make_json(fixed_file, split):
    """
    Writes each book's summary and text to a JSON file named 'fixed_longt5_data_{split}.jsonl'.
    
    Parameters:
        book_data (dict): A dictionary with the source bid as the key and summary and text dictionaries as the value.
        split (str): The name of the split (e.g. 'train', 'val', 'test')
    """
    filtered_data = [data for data in fixed_file.values() if data['text'] != "" and data['summary'] != ""]
    with open(f"booksum_longt5_data_{split}.json", 'w') as f:
        for data in filtered_data:
            json.dump(data, f)
            f.write('\n')
            
def main():
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('split', help='data split to process (train/val/test)', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    fixed_file = get_texts(get_books(args.split), args.split)
    make_json(fixed_file, args.split)
    
if __name__ == "__main__":
    main()