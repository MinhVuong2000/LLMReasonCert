#!/bin/bash
DATASET_LIST="cwq grail_qa"

# gpt-3.5-turbo
for DATASET in $DATASET_LIST; do
    JOB_NAME=gpt-3.5-turbo-fewshot-cot-hint-$DATASET-temp-0.7-p-0.9-consistency-1-is_sc_1
    DATA_PATH=LLMReasoningCert/LLMReasonCert/results/$DATASET/gpt-3.5-turbo/$JOB_NAME/full.jsonl
    python finegrained_gen_cert.py --dat_path $DATA_PATH

    JOB_NAME=gpt-3.5-turbo-fewshot-cot-only-$DATASET-temp-0.7-p-0.9-consistency-1-is_sc_1
    DATA_PATH=LLMReasoningCert/LLMReasonCert/results/$DATASET/gpt-3.5-turbo/$JOB_NAME/full.jsonl
    python finegrained_gen_cert.py --dat_path $DATA_PATH

    JOB_NAME=gpt-3.5-turbo-fewshot-cot-only-$DATASET-temp-0.7-p-0.9-consistency-20-is_sc_4
    DATA_PATH=LLMReasoningCert/LLMReasonCert/results/$DATASET/gpt-3.5-turbo/$JOB_NAME/full.jsonl
    python finegrained_gen_cert.py --dat_path $DATA_PATH
done


# others
MODEL_LIST="Llama-2-70b-chat-hf Qwen-7B-Chat Qwen-14B-Chat Mistral-7B-Instruct-v0.1 vicuna-33b-v1.3"
for DATASET in $DATASET_LIST; do
    for MODEL_NAME in $MODEL_LIST; do
        DATA_PATH=LLMReasoningCert/LLMReasonCert/results/$DATASET/$MODEL_NAME/cot-hint-temp-0.7-p-0.9-is_sc_1/full.jsonl
        python finegrained_gen_cert.py --dat_path $DATA_PATH

        DATA_PATH=LLMReasoningCert/LLMReasonCert/results/$DATASET/$MODEL_NAME/cot-temp-0.7-p-0.9-is_sc_1/full.jsonl
        python finegrained_gen_cert.py --dat_path $DATA_PATH

        DATA_PATH=LLMReasoningCert/LLMReasonCert/results/$DATASET/$MODEL_NAME/cot-temp-0.7-p-0.9-is_sc_4/full.jsonl
        python finegrained_gen_cert.py --dat_path $DATA_PATH
    done
done
