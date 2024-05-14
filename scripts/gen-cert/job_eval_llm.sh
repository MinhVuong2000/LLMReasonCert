#!/bin/bash
DATASET_LIST="cwq grail_qa" #
PROB_THRES=0.6
ENT_THRES=0.5

# gpt-3.5-turbo
for DATASET in $DATASET_LIST; do
    JOB_NAME=gpt-3.5-turbo-fewshot-cot-hint-$DATASET-temp-0.7-p-0.9-consistency-1
    MODE=fewshot-cot-hint
    IS_SC=1
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/gpt-3.5-turbo/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

    JOB_NAME=gpt-3.5-turbo-fewshot-cot-hint-ground-$DATASET-temp-0.7-p-0.9-consistency-1
    MODE=fewshot-cot-hint-ground
    IS_SC=1
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/gpt-3.5-turbo/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

    JOB_NAME=gpt-3.5-turbo-fewshot-cot-only-$DATASET-temp-0.0-p-1.0-consistency-1
    MODE=fewshot-cot-only
    IS_SC=1
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/gpt-3.5-turbo/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

    JOB_NAME=gpt-3.5-turbo-fewshot-cot-only-$DATASET-temp-0.3-p-1.0-consistency-1
    MODE=fewshot-cot-only
    IS_SC=1
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/gpt-3.5-turbo/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

    JOB_NAME=gpt-3.5-turbo-fewshot-cot-only-$DATASET-temp-0.5-p-1.0-consistency-1
    MODE=fewshot-cot-only
    IS_SC=1
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/gpt-3.5-turbo/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

    JOB_NAME=gpt-3.5-turbo-fewshot-cot-only-$DATASET-temp-0.7-p-0.9-consistency-1
    MODE=fewshot-cot-only
    IS_SC=1
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/gpt-3.5-turbo/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

    JOB_NAME=gpt-3.5-turbo-fewshot-cot-only-$DATASET-temp-0.7-p-0.9-consistency-20
    MODE=fewshot-cot-only
    IS_SC=4
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/gpt-3.5-turbo/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $JOB_NAME-$IS_SC generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

    JOB_NAME=gpt-3.5-turbo-fewshot-cot-only-$DATASET-temp-0.7-p-1.0-consistency-1
    MODE=fewshot-cot-only
    IS_SC=1
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/gpt-3.5-turbo/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

    JOB_NAME=gpt-3.5-turbo-fewshot-cot-only-$DATASET-temp-1.0-p-0.95-consistency-1
    MODE=fewshot-cot-only
    IS_SC=1
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/gpt-3.5-turbo/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC
done


# llama 
for DATASET in $DATASET_LIST; do
    for MODEL_SIZE in "70b"; do
        MODEL_NAME=Llama-2-$MODEL_SIZE-chat-hf

        JOB_NAME=cot-hint-temp-0.7-p-0.9
        MODE=fewshot-cot-hint
        IS_SC=1
        DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/$MODEL_NAME/$JOB_NAME/llm_prompt_response.jsonl
        sbatch --job-name $MODEL_NAME-$JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

        JOB_NAME=cot-temp-0.7-p-0.9
        MODE=fewshot-cot-only
        IS_SC=1
        DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/$MODEL_NAME/$JOB_NAME/llm_prompt_response.jsonl
        sbatch --job-name $MODEL_NAME-$JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

        JOB_NAME=cot-temp-0.7-p-0.9
        MODE=fewshot-cot-only
        IS_SC=4
        DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/$MODEL_NAME/$JOB_NAME/llm_prompt_response.jsonl
        sbatch --job-name $MODEL_NAME-$JOB_NAME-$IS_SC generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC
    done
done


# qwen 
for DATASET in $DATASET_LIST; do
    for MODEL_SIZE in "7B" "14B"; do
        MODEL_NAME=Qwen-$MODEL_SIZE-Chat

        JOB_NAME=cot-hint-temp-0.7-p-0.9
        MODE=fewshot-cot-hint
        IS_SC=1
        DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/$MODEL_NAME/$JOB_NAME/llm_prompt_response.jsonl
        sbatch --job-name $MODEL_NAME-$JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

        JOB_NAME=cot-temp-0.7-p-0.9
        MODE=fewshot-cot-only
        IS_SC=1
        DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/$MODEL_NAME/$JOB_NAME/llm_prompt_response.jsonl
        sbatch --job-name $MODEL_NAME-$JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

        JOB_NAME=cot-temp-0.7-p-0.9
        MODE=fewshot-cot-only
        IS_SC=4
        DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/$MODEL_NAME/$JOB_NAME/llm_prompt_response.jsonl
        sbatch --job-name $MODEL_NAME-$JOB_NAME-$IS_SC generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC
    done
done


# mistral 
for DATASET in $DATASET_LIST; do
    MODEL_NAME=Mistral-7B-Instruct-v0.1

    JOB_NAME=cot-hint-temp-0.7-p-0.9
    MODE=fewshot-cot-hint
    IS_SC=1
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/$MODEL_NAME/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $MODEL_NAME-$JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

    JOB_NAME=cot-temp-0.7-p-0.9
    MODE=fewshot-cot-only
    IS_SC=1
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/$MODEL_NAME/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $MODEL_NAME-$JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

    JOB_NAME=cot-temp-0.7-p-0.9
    MODE=fewshot-cot-only
    IS_SC=4
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/$MODEL_NAME/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $MODEL_NAME-$JOB_NAME-$IS_SC generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC
done


# vicuna 
for DATASET in $DATASET_LIST; do
    MODEL_NAME=vicuna-33b-v1.3

    JOB_NAME=cot-hint-temp-0.7-p-0.9
    MODE=fewshot-cot-hint
    IS_SC=1
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/$MODEL_NAME/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $MODEL_NAME-$JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

    JOB_NAME=cot-temp-0.7-p-0.9
    MODE=fewshot-cot-only
    IS_SC=1
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/$MODEL_NAME/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $MODEL_NAME-$JOB_NAME generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC

    JOB_NAME=cot-temp-0.7-p-0.9
    MODE=fewshot-cot-only
    IS_SC=4
    DATA_PATH=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/data/$DATASET/$MODEL_NAME/$JOB_NAME/llm_prompt_response.jsonl
    sbatch --job-name $MODEL_NAME-$JOB_NAME-$IS_SC generative_cert/scripts/eval_llm.sh $DATASET $MODE $DATA_PATH $PROB_THRES $ENT_THRES $IS_SC
done
