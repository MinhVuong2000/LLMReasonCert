#!/bin/bash
#SBATCH --job-name=fewshot-cot
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1               # node count
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1    # total number of tasks per node
#SBATCH --mem-per-cpu=50000
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trang.vu1@monash.edu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ROOT_DIR=/data/others/trang/cert-reasoning
module load anaconda/anaconda3
module load cudnn/8.5.0.96-11.7-gcc-8.5.0-l5kw6yn
module load cuda/11.7.0-gcc-8.5.0-xcmnp4n
source activate $ROOT_DIR/env

export TMPDIR=$ROOT_DIR/tmp
#export HF_HOME=/data/others/lluo/huggingface/
export HF_HOME=$ROOT_DIR/huggingface
export HUGGINGFACE_HUB_CACHE=$ROOT_DIR/huggingface
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1

SRC_DIR=$ROOT_DIR/LLMReasonCert
DATA_DIR=$ROOT_DIR/data/{}/gpt-3.5-turbo/test/splitted_ground_truth_paths.json
OUT_DIR=$ROOT_DIR/output
MODEL_NAME=$1
HF_MODEL_NAME=$2
DATASET=$3
TEMP=0.7
TOPP=0.9
EXP_NAME=cot-temp-$TEMP-p-$TOPP
NUM_RETURN_SEQUENCE=4
MODE="fewshot-cot-only"

echo "============================================================="
echo "Running generation with temprature sampling"
echo " * DATA_DIR            = "$DATA_DIR
echo " * DATASET             = "$DATASET
echo " * OUT_DIR             = "$OUT_DIR
echo " * MODEL_NAME          = "$MODEL_NAME
echo " * HF_MODEL_NAME       = "$HF_MODEL_NAME
echo " * EXP_NAME            = "$EXP_NAME
echo " * TEMPERATURE         = "$TEMP
echo " * TOPP                = "$TOPP
echo " * NUM_RETURN_SEQUENCE = "$NUM_RETURN_SEQUENCE
echo " * BATCH_SIZE          = "$BATCH_SIZE
echo " * MODE                = "$MODE
echo "============================================================="

python3 $SRC_DIR/llm_generation.py \
    --model_name $MODEL_NAME --model_path $HF_MODEL_NAME \
    --dataset $DATASET --mode $MODE \
    --in_dir $DATA_DIR \
    --out_dir $OUT_DIR \
    --temperature $TEMP --top_p $TOPP \
    --exp_name $EXP_NAME --num_return_sequences $NUM_RETURN_SEQUENCE
