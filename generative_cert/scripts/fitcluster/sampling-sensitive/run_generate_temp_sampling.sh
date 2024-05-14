#!/bin/bash
#SBATCH --job-name=llm_generate
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
export HF_HOME=$ROOT_DIR/huggingface
export HUGGINGFACE_HUB_CACHE=$ROOT_DIR/huggingface


SRC_DIR=$ROOT_DIR/LLMReasonCert
DATA_DIR=$ROOT_DIR/data/{}/gpt-3.5-turbo/test/splitted_ground_truth_paths.json
OUT_DIR=$ROOT_DIR/output
HF_MODEL_NAME=$1
DATASET=$2
TEMP=$3
EXP_NAME=sampling-temp-$TEMP
PROMT_TEMPLATE_DIR=$SRC_DIR/srcs/conf/generation
DEVICE=cuda

echo "============================================================="
echo "Running generation with temprature sampling"
echo " * DATA_DIR           = "$DATA_DIR
echo " * OUT_DIR            = "$OUT_DIR
echo " * HF_MODEL_NAME      = "$HF_MODEL_NAME
echo " * PROMT_TEMPLATE_DIR = "$PROMT_TEMPLATE_DIR
echo " * DEVICE             = "$DEVICE
echo " * EXP_NAME           = "$EXP_NAME
echo " * TEMPERATURE        = "$TEMP
echo " * TOPK               = "$TOPK
echo " * TOPP               = "$TOPP
echo "============================================================="

python $SRC_DIR/srcs/main.py \
    --HF_model_name $HF_MODEL_NAME \
    --dataset $DATASET \
    --in_dir $DATA_DIR \
    --out_dir $OUT_DIR \
    --prompt_template_dir $PROMT_TEMPLATE_DIR \
    --device $DEVICE \
    --temperature $TEMP --exp_name $EXP_NAME
