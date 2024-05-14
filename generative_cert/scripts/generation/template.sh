#!/bin/bash
#SBATCH --job-name=llm_generate
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1               # node count
# SBATCH --ntasks=1
# SBATCH --ntasks-per-node=1    # total number of tasks per node  
#SBATCH --mem-per-cpu=50000
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # number of gpus
# SBATCH --qos=normal
#SBATCH --mail-type=BEGIN,END,FAIL
# SBATCH --mail-user=minhvuong160620@gmail.com
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load python/3.8.5

ROOT_DIR=LLMReasoningCert/
export HUGGINGFACE_HUB_CACHE=$ROOT_DIR/huggingface
export HF_HOME=$ROOT_DIR/huggingface
export PYTHONPATH="${PYTHONPATH}:LLMReasoningCert/LLMReasonCert"

source $ROOT_DIR/envs/vuongntm/bin/activate  
cd $ROOT_DIR/LLMReasonCert

HF_MODEL_NAME=$1
DATASET=$3
DATA_DIR=LLMReasoningCert/data/{}/gpt-3.5-turbo/test/splitted_ground_truth_paths.json
OUT_DIR=LLMReasoningCert/data
PROMT_TEMPLATE_DIR=srcs/conf/generation
DEVICE=cuda
TEMP=1.0
TOPK=$4
MODE=$2
NUM_RETURN_SEQUENCE=$5
BATCH_SIZE=$6
EXP_NAME=sampling-topk-$TOPK


echo "============================================================="
echo "Running generation with topk sampling"
echo " * DATA_DIR            = "$DATA_DIR
echo " * OUT_DIR             = "$OUT_DIR
echo " * HF_MODEL_NAME       = "$HF_MODEL_NAME
echo " * PROMT_TEMPLATE_DIR  = "$PROMT_TEMPLATE_DIR
echo " * MODE                = "$MODE
echo " * DEVICE              = "$DEVICE
echo " * EXP_NAME            = "$EXP_NAME
echo " * TEMPERATURE         = "$TEMP
echo " * TOPK                = "$TOPK
echo " * TOPP                = "$TOPP
echo " * NUM_RETURN_SEQUENCE = "$NUM_RETURN_SEQUENCE
echo " * BATCH_SIZE          = "$BATCH_SIZE
echo "============================================================="


python llm_generation.py \
    --model_name $HF_MODEL_NAME \
    --dataset $DATASET \
    --in_dir $DATA_DIR \
    --out_dir $OUT_DIR \
    --temperature $TEMP \
    --exp_name $EXP_NAME \
    # --top_k $TOPK \
    --mode $MODE \
    # --model_path $HF_MODEL_NAME \
    # --dtype fp16 \
    # --quant none \
    --num_return_sequences $NUM_RETURN_SEQUENCE \
    # --batch_size $BATCH_SIZE \
    --run_sample