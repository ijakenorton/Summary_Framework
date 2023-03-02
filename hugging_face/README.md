

Run this to train the model
run from the root of transformers
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

The evaluation script is currently in