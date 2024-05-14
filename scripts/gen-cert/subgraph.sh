#!/bin/bash
# SBATCH --job-name=get_subgraph
# SBATCH --account=da34
# SBATCH --time=04:00:00
# SBATCH --nodes=1                # node count
# SBATCH --ntasks=1
# SBATCH --ntasks-per-node=1    # total number of tasks per node
# SBATCH --cpus-per-task=16      
# SBATCH --gres=None         # number of gpus
# SBATCH --mem-per-cpu=40960
# SBATCH --partition=fit
# SBATCH --qos=fit
# SBATCH --mail-type=BEGIN,END,FAIL
# SBATCH --mail-user=minhvuong160620@gmail.com
# SBATCH --output=%x-%j.out
# SBATCH --error=%x-%j.err

module load python/3.8.5
ROOT_DIR='LLMReasoningCert'
source $ROOT_DIR/envs/vuongntm/bin/activate

export HUGGINGFACE_HUB_CACHE='envs/huggingface/'
export HF_HOME='envs/huggingface/'

cd $ROOT_DIR/LLMReasonCert
srun --jobid $SLURM_JOBID bash -c 'python preprocess_data.py'
