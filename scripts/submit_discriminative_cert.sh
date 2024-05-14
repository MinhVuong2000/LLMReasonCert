MODEL_LIST="gpt-3.5-turbo"
MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
N_PROCESS=5

for MODEL_NAME in $MODEL_LIST; do
    for MODE in $MODE_LIST; do
        echo "Submitting $MODEL $MODE"
        sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --n 1 submit_CPU_job.sh \
        "python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --n $N_PROCESS"
    done
done


MODEL_LIST="llama2-7B-chat-hf"
MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
for MODEL_NAME in $MODEL_LIST; do
    for MODE in $MODE_LIST; do
        echo "Submitting $MODEL_NAME $MODE"
        sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 submit_1-GPU_job.sh \
        "python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path $MODEL_PATH"
        sleep 10
    done
done

MODEL_LIST="llama2-13B-chat-hf"
MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
MODEL_PATH="meta-llama/Llama-2-13b-chat-hf"
for MODEL_NAME in $MODEL_LIST; do
    for MODE in $MODE_LIST; do
        echo "Submitting $MODEL_NAME $MODE"
        sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 --exclude="node[01-05]" submit_1-GPU_job.sh \
        "TMPDIR=/home/lluo/projects/LLMReasonCert/tmp TEMP=/home/lluo/projects/LLMReasonCert/tmp TMP=/home/lluo/projects/LLMReasonCert/tmp python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path $MODEL_PATH"
    done
done

MODEL_LIST="llama2-70B-chat-hf"
MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
MODEL_PATH="meta-llama/Llama-2-70b-chat-hf"
for MODEL_NAME in $MODEL_LIST; do
    for MODE in $MODE_LIST; do
        echo "Submitting $MODEL_NAME $MODE"
        sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 submit_1-A100_job.sh \
        "python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path $MODEL_PATH --quant 8bit"
    done
done


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

MODEL_LIST="Mistral-7B-Instruct-v0.1"
MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.1"
for MODEL_NAME in $MODEL_LIST; do
    for MODE in $MODE_LIST; do
        echo "Submitting $MODEL_NAME $MODE"
        sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 submit_1-GPU_job.sh \
        "conda activate py310_hf_350" \
        "python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path $MODEL_PATH"
    done
done

MODEL_LIST="Qwen-14B-Chat"
MODE_LIST="zero-shot zero-shot-cot few-shot few-shot-cot"
MODEL_PATH="Qwen/Qwen-14B-Chat"
for MODEL_NAME in $MODEL_LIST; do
    for MODE in $MODE_LIST; do
        echo "Submitting $MODEL_NAME $MODE"
        sbatch --job-name ${MODEL_NAME}-${MODE} --cpus-per-task 1 --ntasks 1 --exclude="node[01-05]" submit_1-GPU_job.sh \
        "conda activate py310_hf_350" \
        "TMPDIR=/home/lluo/projects/LLMReasonCert/tmp python discriminative-cert.py --model_name $MODEL_NAME --mode $MODE --model_path $MODEL_PATH"
    done
done