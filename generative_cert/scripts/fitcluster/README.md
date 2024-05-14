# Installation
```bash
module load anaconda/anaconda3
conda create --p env python=3.10
source activate ./env
conda install pip

module load cudnn/8.5.0.96-11.7-gcc-8.5.0-l5kw6yn
module load cuda/11.7.0-gcc-8.5.0-xcmnp4n

#python3 -m venv env
#source env/bin/activate
pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

cd LLMReasonCert; pip install -r requirements.txt
```

# Experiments
Setup environment:
- Add `.env` file with your OPENAI key and HF_TOKEN. Please check the `.env.example` for an example
## LLMs
We will evaluate on the following LLMs
```
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-chat-hf
- meta-llama/Llama-2-70b-chat-hf
- mistralai/Mistral-7B-Instruct-v0.1
- Qwen/Qwen-14B-Chat
- lmsys/vicuna-33b-v1.3

TODO: llama2-instruct
```

## Submit jobs on Fitcluster
Note the model_name for different models
- For qwen family, set `MODEL_NAME=qwen`
- For mistra family, set `MODEL_NAME=mistral`
- For vicuna family, set `MODEL_NAME=vicuna`
- For llama family, set `MODEL_NAME=llama`

### Fewshot-CoT
```bash
MODEL_NAME=llama
HF_MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
MODEL_PREFIX=llama2-7b
for DATASET in cwq grail_qa; do
    sbatch --job-name=$MODEL_PREFIX-$DATASET run-fewshot-cot.sh $MODEL_NAME $HF_MODEL_NAME $DATASET
done
```

### Fewshot-CoT-with-hint
```bash
MODEL_NAME=llama
HF_MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
MODEL_PREFIX=llama2-7b
for DATASET in cwq grail_qa; do
    sbatch --job-name=hint-$MODEL_PREFIX-$DATASET run-fewshot-cot-with-hint.sh $MODEL_NAME  $HF_MODEL_NAME $DATASET
done
```
