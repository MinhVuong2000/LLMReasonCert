#!/bin/bash

MODEL_NAME=llama
HF_MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
MODEL_PREFIX=llama2-7b
for DATASET in cwq webqsp grail_qa; do
    sbatch --job-name=$MODEL_PREFIX-$DATASET run-fewshot-cot.sh $MODEL_NAME $HF_MODEL_NAME $DATASET
done