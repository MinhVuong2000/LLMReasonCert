#!/bin/bash
#SBATCH --job-name=llm_generate
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1               # node count
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1    # total number of tasks per node
#SBATCH --mem-per-cpu=50000
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minhvuong160620@gmail.com
#SBATCH --output=LLMReasoningCert/slum/llm_generation/%x-%j.out
#SBATCH --error=LLMReasoningCert/slum/llm_generation/%x-%j.err

module load python/3.8.5

ROOT_DIR='LLMReasoningCert/'
export HUGGINGFACE_HUB_CACHE=$ROOT_DIR/huggingface
export HF_HOME=$ROOT_DIR/huggingface

source $ROOT_DIR/envs/vuongntm/bin/activate  
cd $ROOT_DIR/LLMReasonCert

MODEL_NAME=$1
MODE=$2
DATASET=$3
N_PROCESS=$4
TEMP=$5
TOPP=$6
N_SEQ=$7
EXP_NAME=$MODEL_NAME-$MODE-$DATASET-temp-$TEMP-p-$TOPP-consistency-$N_SEQ
echo "============================================================="
echo "Running generation with temprature sampling"
echo " * MODEL_NAME          = "$MODEL_NAME
echo " * MODE                = "$MODE
echo " * DATASET             = "$DATASET
echo " * EXP_NAME            = "$EXP_NAME
echo " * TEMPERATURE         = "$TEMP
echo " * TOPP                = "$TOPP
echo " * NUM_RETURN_SEQUENCE = "$N_SEQ
echo "============================================================="

python llm_generation.py \
    --exp_name $EXP_NAME \
    --model_name $MODEL_NAME \
    --mode $MODE \
    --dataset $DATASET \
    --n $N_PROCESS \
    --temperature $TEMP \
    --top_p $TOPP \
    --num_return_sequences $N_SEQ 
    