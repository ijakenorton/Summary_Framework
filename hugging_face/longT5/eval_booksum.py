import evaluate
import os
import argparse
from tqdm.auto import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from datasets import load_dataset
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
parser = argparse.ArgumentParser(description='Test model checkpoint for test data')
parser.add_argument('--test_data_path', help='path to test data, eg ../booksum_longt5_data_test.json', default="../booksum_longt5_data_test.json")
parser.add_argument('--model_path', help='path to model checkpoint, ../../tmp/tst-summarization/checkpoint-7000/pytorch_model.bint', default="../../tmp/tst-summarization/checkpoint-7000/pytorch_model.bin")
args = parser.parse_args()
dataset = load_dataset("json", data_files=args.test_data_path)
model = LongT5ForConditionalGeneration.from_pretrained("/home/norja159/Documents/tmp/tst-summarization/checkpoint-7000/pytorch_model.bin",config='/home/norja159/Documents/tmp/tst-summarization/checkpoint-7000/config.json').to('cuda')

tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
def generate_answers(batch):
    # print('batch length ->> ',len(batch['text'][0])) 
    current = 2048
    pbar = tqdm(total=len(batch['text'][0])/current)
    current_slice = 0
    
    for i in range(current, len(batch['text'][0]), current):  
        print('current_slice ->>>', current_slice)
        print('i >>>> ',i)
        current_batch = batch["text"][0][slice(current_slice,i)]
        inputs_dict = tokenizer(
            current_batch, max_length=current, padding=False, truncation=False, return_tensors="pt"
        )
        input_ids = inputs_dict.input_ids.to("cuda")
        attention_mask = inputs_dict.attention_mask.to("cuda")
        output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=512, num_beams=4, repetition_penalty=1.5)
        if "predicted_summary" not in batch.keys():
            batch["predicted_summary"] = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            print('current batch ->>',current_batch)
            print('==' * 10)
            print(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
            print('==' * 10)
            current_slice = i
            continue
        batch["predicted_summary"][0] += (tokenizer.batch_decode(output_ids, skip_special_tokens=True))[0]
        print('current batch ->>',current_batch)
        print('==' * 10)
        print(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        print('==' * 10)
        current_slice = i
        pbar.update()
        
    # print(batch["predicted_summary"])
    pbar.close()
    return batch


result = dataset.map(generate_answers, batched=True, batch_size=1)
rouge = evaluate.load("rouge")
print(rouge.compute(predictions=result["train"]["predicted_summary"], references=result["train"]["summary"]))