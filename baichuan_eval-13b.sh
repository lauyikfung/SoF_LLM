#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python eval_baichuan.py --model_type baichuan \
--model_name_or_path /your/path/to/baichuan/model \
--tokenizer_name_or_path /your/path/to/baichuan/model \
--train_file_dir /your/training/directory \
--validation_file_dir /your/valid/directory \
--peft_path /your/path/to/baichuan/model \
--do_eval \
--eval_step 100 \
--use_peft False \
--flash_attn False \
--lr_scheduler_type cosine \
--warmup_ratio 0.05 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--output_dir /your/path/for/output/baichuan/model \
--overwrite_output_dir \
--max_grad_norm 1.0 \
--save_steps 1000 \
--logging_steps 5 \
--seed 42 \
--template_name baichuan \
--run_name baichuan