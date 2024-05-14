#!/bin/bash
#SBATCH --job-name=certify_fact
# SBATCH --account=da34
#SBATCH --time=01:00:00
#SBATCH --nodes=1               # node count
# SBATCH --ntasks=1
# SBATCH --ntasks-per-node=1    # total number of tasks per node  
# SBATCH --gres=1         # number of gpus
#SBATCH --mem-per-cpu=50000
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
# SBATCH --qos=normal
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=minhvuong160620@gmail.com

module load python/3.8.5

ROOT_DIR='LLMReasoningCert/'
HUGGINGFACE_HUB_CACHE=$ROOT_DIR/envs/huggingface
HF_HOME=$ROOT_DIR/envs/huggingface

source $ROOT_DIR/envs/vuongntm/bin/activate  
cd $ROOT_DIR/LLMReasonCert
python certify_fact.py --prob_thres 70 #--dataset cwq
