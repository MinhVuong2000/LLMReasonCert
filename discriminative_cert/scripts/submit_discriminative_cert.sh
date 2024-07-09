# DATA_PATH="data/cwq_test_res.csv"
DATA_PATH="data/multi_hop_grailqa.csv"
MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"


MODEL_NAME="gpt-3.5-turbo"
MODEL_PATH="None"
N_PROCESS=5
QUANT=none

# MODEL_NAME="llama2-7B-chat-hf"
# MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
# N_PROCESS=1
# QUANT=none

# MODEL_NAME="llama2-13B-chat-hf"
# MODEL_PATH="meta-llama/Llama-2-13b-chat-hf"
# N_PROCESS=1
# QUANT=none

# MODEL_NAME="Mistral-7B-Instruct-v0.1"
# MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.1"
# N_PROCESS=1
# QUANT=none

# MODEL_NAME="Qwen-7B-Chat"
# MODEL_PATH="Qwen/Qwen-7B-Chat"
# N_PROCESS=1
# QUANT=none

# MODEL_NAME="Qwen-14B-Chat"
# MODEL_PATH="Qwen/Qwen-14B-Chat"
# N_PROCESS=1
# QUANT=none


# MODEL_NAME="vicuna-33b-v1.3"
# MODEL_PATH="lmsys/vicuna-33b-v1.3"
# N_PROCESS=1
# QUANT=none

# MODEL_NAME="llama2-70B-chat-hf"
# MODEL_PATH="meta-llama/Llama-2-70b-chat-hf"
# N_PROCESS=1
# QUANT=4bit


for DATA in $DATA_PATH; do
    for MODE in $MODE_LIST; do
        echo "Running $DATA $MODEL_NAME $MODE"
        python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path ${MODEL_PATH} --n $N_PROCESS --data_path $DATA --quant $QUANT
    done
done



















# DATA_PATH=data/multi_hop_grailqa.csv
# DATA_PATH=data/cwq_test_res.csv

# MODEL_LIST="gpt-3.5-turbo"
# MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
# N_PROCESS=5
# for MODEL_NAME in $MODEL_LIST; do
#     FIRST=True
#     for MODE in $MODE_LIST; do
#         echo "Submitting $MODEL $MODE"
#         sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 submit_CPU_job.sh \
#         "python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --n $N_PROCESS --data_path ${DATA_PATH} --wandb"
#         if [ "$FIRST" = True ]; then
#             sleep 10
#             FIRST=False
#         fi
#     done
# done


# MODEL_LIST="llama2-7B-chat-hf"
# MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
# MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
# for MODEL_NAME in $MODEL_LIST; do
#     FIRST=True
#     for MODE in $MODE_LIST; do
#         echo "Submitting $MODEL_NAME $MODE"
#         sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 submit_1-GPU_job.sh \
#         "python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path $MODEL_PATH --data_path ${DATA_PATH} --wandb"
#         if [ "$FIRST" = True ]; then
#             sleep 10
#             FIRST=False
#         fi
#     done
# done

# MODEL_LIST="llama2-13B-chat-hf"
# MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
# MODEL_PATH="meta-llama/Llama-2-13b-chat-hf"
# for MODEL_NAME in $MODEL_LIST; do
#     FIRST=True
#     for MODE in $MODE_LIST; do
#         echo "Submitting $MODEL_NAME $MODE"
#         sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 --exclude="node[01-05]" submit_1-GPU_job.sh \
#         "TMPDIR=/home/lluo/projects/LLMReasonCert/tmp TEMP=/home/lluo/projects/LLMReasonCert/tmp TMP=/home/lluo/projects/LLMReasonCert/tmp python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path $MODEL_PATH --data_path ${DATA_PATH} --wandb"
#         if [ "$FIRST" = True ]; then
#             sleep 10
#             FIRST=False
#         fi
#     done
# done

# MODEL_LIST="llama2-70B-chat-hf"
# MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
# MODEL_PATH="meta-llama/Llama-2-70b-chat-hf"
# for MODEL_NAME in $MODEL_LIST; do
#     FIRST=True
#     for MODE in $MODE_LIST; do
#         echo "Submitting $MODEL_NAME $MODE"
#         sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 submit_1-A100_job.sh \
#         "python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path $MODEL_PATH --quant 8bit --data_path ${DATA_PATH} --wandb"
#         if [ "$FIRST" = True ]; then
#             sleep 10
#             FIRST=False
#         fi
#     done
# done


# MODEL_LIST="llama2-70B-chat-hf_fp16"
# MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
# MODEL_PATH="meta-llama/Llama-2-70b-chat-hf"
# for MODEL_NAME in $MODEL_LIST; do
#     for MODE in $MODE_LIST; do
#         echo "Submitting $MODEL_NAME $MODE"
#         sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 submit_2-A100_job.sh \
#         "python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path $MODEL_PATH"
#     done
# done

# MODEL_LIST="Mistral-7B-Instruct-v0.1"
# MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
# MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.1"
# for MODEL_NAME in $MODEL_LIST; do
#     FIRST=True
#     for MODE in $MODE_LIST; do
#         echo "Submitting $MODEL_NAME $MODE"
#         sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 submit_1-GPU_job.sh \
#         "conda activate py310_hf_350" \
#         "python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path $MODEL_PATH --data_path ${DATA_PATH} --wandb"
#         if [ "$FIRST" = True ]; then
#             sleep 10
#             FIRST=False
#         fi
#     done
# done

# MODEL_LIST="Qwen-7B-Chat"
# MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
# MODEL_PATH="Qwen/Qwen-7B-Chat"
# for MODEL_NAME in $MODEL_LIST; do
#     FIRST=True
#     for MODE in $MODE_LIST; do
#         echo "Submitting $MODEL_NAME $MODE"
#         sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 --exclude="node[01-06]" submit_1-GPU_job.sh \
#         "conda activate py310_hf_350" \
#         "TMPDIR=/home/lluo/projects/LLMReasonCert/tmp python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path $MODEL_PATH --data_path ${DATA_PATH} --wandb"
#         if [ "$FIRST" = True ]; then
#             sleep 10
#             FIRST=False
#         fi
#     done
# done

# MODEL_LIST="Qwen-14B-Chat"
# MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
# MODEL_PATH="Qwen/Qwen-14B-Chat"
# for MODEL_NAME in $MODEL_LIST; do
#     FIRST=True
#     for MODE in $MODE_LIST; do
#         echo "Submitting $MODEL_NAME $MODE"
#         sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 --exclude="node[01-06]" submit_1-GPU_job.sh \
#         "conda activate py310_hf_350" \
#         "TMPDIR=/home/lluo/projects/LLMReasonCert/tmp python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path $MODEL_PATH --data_path ${DATA_PATH} --wandb"
#         if [ "$FIRST" = True ]; then
#             sleep 10
#             FIRST=False
#         fi
#     done
# done

# MODEL_LIST="vicuna-33b-v1.3"
# MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
# MODEL_PATH="lmsys/vicuna-33b-v1.3"
# for MODEL_NAME in $MODEL_LIST; do
#     FIRST=True
#     for MODE in $MODE_LIST; do
#         echo "Submitting $MODEL_NAME $MODE"
#         sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 submit_1-A100_job.sh \
#         "conda activate py310_hf_350" \
#         "TMPDIR=/home/lluo/projects/LLMReasonCert/tmp python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path $MODEL_PATH --data_path ${DATA_PATH} --wandb"
#         if [ "$FIRST" = True ]; then
#             sleep 10
#             FIRST=False
#         fi
#     done
# done
