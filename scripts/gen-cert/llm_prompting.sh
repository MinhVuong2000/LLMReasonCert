#!/bin/bash
MODEL_NAME=gpt-3.5-turbo
N_PROCESS=8

## consistency
TOPP=0.9
TEMP=0.7
MODE=fewshot-cot-only
N_SEQ=20
for DATASET in cwq grail_qa; do
  sbatch generative_cert/scripts/generation/chatgpt_generation.sh $MODEL_NAME $MODE $DATASET $N_PROCESS $TEMP $TOPP $N_SEQ
done

## temprerature topp
TOPP=0.9
TEMP=0.7
N_SEQ=1
for DATASET in cwq grail_qa; do
  for MODE in "fewshot-cot-only" "fewshot-cot-hint" "fewshot-cot-hint-ground"; do
    sbatch generative_cert/scripts/generation/chatgpt_generation.sh $MODEL_NAME $MODE $DATASET $N_PROCESS $TEMP $TOPP $N_SEQ
  done
done