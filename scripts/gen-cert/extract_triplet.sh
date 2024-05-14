#!/bin/bash
#SBATCH --job-name=extract_triplet-skip_unknown_ent-only_test
# SBATCH --account=da34
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1               # node count
# SBATCH --ntasks=1
# SBATCH --ntasks-per-node=1    # total number of tasks per node  
# SBATCH --gres=0         # number of gpus
#SBATCH --mem-per-cpu=50000
#SBATCH --cpus-per-task=2
# SBATCH --partition=gpu
# SBATCH --qos=normal
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minhvuong160620@gmail.com
#SBATCH --output=LLMReasoningCert/slum/%x-%j.out
#SBATCH --error=LLMReasoningCert/slum/%x-%j.err

module load python/3.8.5

ROOT_DIR='LLMReasoningCert/'
export HUGGINGFACE_HUB_CACHE=$ROOT_DIR/huggingface
export HF_HOME=$ROOT_DIR/huggingface

export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}/LLMReasonCert"

source $ROOT_DIR/envs/vuongntm/bin/activate  
cd $ROOT_DIR/LLMReasonCert
DATASET=$1
python extract_triplet/extract_triplet.py --dataset $DATASET --create_db ###--query 'Francis Avent Gumm was born in Tennessee'
