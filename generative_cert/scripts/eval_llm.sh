#!/bin/bash
#SBATCH --job-name=llm_generate
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1               # node count
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1    # total number of tasks per node
#SBATCH --mem-per-cpu=50000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minhvuong160620@gmail.com
#SBATCH --output=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/slum/llm_cert/%x-%j.out
#SBATCH --error=/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/slum/llm_cert/%x-%j.err

module load python/3.8.5

ROOT_DIR='/home/xvuthith/da33_scratch/lluo/LLMReasoningCert/'
export HUGGINGFACE_HUB_CACHE=$ROOT_DIR/huggingface
export HF_HOME=$ROOT_DIR/huggingface

source $ROOT_DIR/envs/vuongntm/bin/activate  
cd $ROOT_DIR/LLMReasonCert

DATASET=$1
MODE=$2
DATA_PATH=$3
PROB_THRES=$4
ENT_THRES=$5
IS_SC=$6

echo "============================================================="
echo "Running certification of LLM"
echo " * RAW_DATA_PATH       = "$DATA_PATH
echo " * MODE                = "$MODE
echo " * DATASET             = "$DATASET
echo " * PROB_THRES          = "$PROB_THRES
echo " * ENT_THRES           = "$ENT_THRES
echo " * SELF_CONSISTENCY    = "$IS_SC
echo "============================================================="


python generative-cert.py \
    --dataset $DATASET \
    --mode $MODE \
    --raw_dat_path $DATA_PATH \
    --prob_thres $PROB_THRES \
    --ent_thres $ENT_THRES \
    --is_sc $IS_SC
