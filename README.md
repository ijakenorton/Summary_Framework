This repo contains two long summization frameworks
TODO
Add download links to model checkpoints
To install the dependencies required for memsum, the README in the top of the memsum directory is well explained and detailed. However if you run into trouble with the pytorch versioning I used this as the gpus I was using required a higher version
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```
Or you can install from memsum.yml provided
```
conda env create -f memsum.yml
conda activate memsum
```
To install the dependencies for huggingface & longt5 
```
conda env create -f longt5.yml
conda activate huggingface
cd hugging_face/transformers
pip install -e .
```
***Data***
The dataset used in both models is from booksum, it is the human written concatenated section summaries of each book for each source as the text to summarize, the gold summary for it to be compared with is the human written whole book summary for each source. Some of the summaries from booksum have been removed as these summaries had something missing from the dataset. This is due to the web scraping links expiring.
This data is created for memsum using ```make_memsum_data.py```and for longt5 ```make_longt5_data.py``` Some of the paths to files will need to be changed according to where you store the booksum data. 
**Memsum**
As created from ```make_memsum_data.py```, memsum takes data as a jsonl, each json containing two keys "text" and "summary" both containing a list of sentences.
The memsum readme and example jupiter notebook (./MemSum/Data_processing_training_and_testing_for_MemSum.ipynb) provide a good guide in preparing data for usuage in the model and testing, additionally the data I used for creating the model is provided in the repo. However, if you have new data which is also split into the correct jsonl format, then run ```high_rouge.py``` with the path of input path and output path
## Download pretrained word embedding

MemSUM used the glove embedding (200dim), with three addition token embeddings for bos eos pad, etc.

You can download the word embedding (a folder named glove/) used in this work:

https://drive.google.com/drive/folders/1lrwYrrM3h0-9fwWCOmpRkydvmF6hmvmW?usp=sharing

and put the folder under the model/ folder. 

Or you can do it using the code below:

Make sure the structure looks like:

1. MemSum/model/glove/unigram_embeddings_200dim.pkl
2. MemSum/model/glove/vocabulary_200dim.pkl


If not, you can change manually
**Longt5**
As created from ```make_memsum_data.py```, memsum takes data as a jsonl, each json containing two keys "text" and "summary" containing the text as one long string.

***Training***
**Memsum**
An example of training the memsum model, this is the parameters for the latest model, bear in mind that though this goes for 100 epochs, I have found that the model taken from ~35 epochs performs the best, or at batch 7301
```
cd ./MemSum/src/MemSum_Full
python train.py -training_corpus_file_name ../../data/booksum/fixed_memsum_data_list_train_labelled.jsonl -validation_corpus_file_name ../../data/booksum/fixed_memsum_data_list_val.jsonl -model_folder ../../model/MemSum_Full/booksum/200dim/run4/ -log_folder ../../log/MemSum_Full/booksum/200dim/run4/ -vocabulary_file_name ../../model/glove/vocabulary_200dim.pkl -pretrained_unigram_embeddings_file_name ../../model/glove/unigram_embeddings_200dim.pkl -max_seq_len 100 -max_doc_len 500 -num_of_epochs 100 -save_every 1000 -n_device 2 -batch_size_per_device 1 -max_extracted_sentences_per_document 55 -moving_average_decay 0.999 -p_stop_thres 0.6
```
**Longt5**
An example of a training script for the longt5 model. I have done less testing, from the evaluation score it seems like epoch 38 is the best, however I don't think that in the current state of the training data the model copes well with the length of input. The average length of input for the booksum training data is ~66000 tokens, which is much larger than the recommended 16000, so I believe there is a lot of truncation happening, which leads to a poorly trained model. 
```
cd ./hugging_face/transformers
python -m torch.distributed.launch \
    --nproc_per_node 2 examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google/long-t5-tglobal-base \
    --train_file ../booksum_longt5_data_train.json \
    --validation_file ../booksum_longt5_data_val.json \
    --do_train \
    --do_eval \
    --num_beams 4 \
    --source_prefix "summarize: " \
    --output_dir ~/tmp/fixed-tst-summarization \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --overwrite_output_dir \
    --num_train_epochs 100 \
    --predict_with_generate \
    --evaluation_strategy epoch
```
***Evaluation***
Both evaluation scripts are still quite hardcoded, though they should work fine as they are for the current dataset once you have a model checkpoint or have trained your own model
**Memsum**
To test the model
```
cd ./MemSum
python evaluate.py
```
evaluate.py should work out the gate with the current inputs, providing you have the default model checkpoint. It runs tests on the booksum test data and outputs rouge 1,2,l scores. The best score is currently from checkpoint batch 7301 with a max_extracted_sentences_per_document = 55

**Longt5**
To test the model
```
cd ./hugging_face/longt5
python eval_booksum.py
```
This testing script is very much in a work in progress, as the model doesn't handle the input size of the data it by default will truncate any excess, which makes it quite pointless. So this script splits the input data by currently an arbitrary amount (2048), summarises each chunk, then concatenates them back together. This gives a much better result than the default, but still not fantastic, and does not play to the strengths of the model.