A Guide for LLM Generation
---

## Run step by step:
1. Move to the main working directory
```bash
cd LLMReasoningCert/LLMReasonCert
```
2. Modify bash file at `./scripts/generation/template.sh`. \
e.g. `sbatch generative_cert/scripts/generation/template.sh mistralai/Mistral-7B-Instruct-v0.1 fewshot-cot-only cwq 50 1 4`
arguments can be found in `generative_cert/llm_generation.py` and `llms/base_hf_causal_model.py`

3. Run a few samples to make sure output is OK \
3.1.
`--run_sample` \
Default is DATASET can be `all`, mean ['cwq','FreebaseQA','FreebaseQA]. You can only generate 1 out of 3 datasets by change row DATASET
3.2. Check Output results to make sure output is OK (3 files)\
a. `../data/cwq/{model_name}/test/llm_prompt_response.jsonl`\
b. `../data/FreebaseQA/{model_name}/test/llm_prompt_response.jsonl`\
b. `../data/grail_qa/{model_name}/test/llm_prompt_response.jsonl`\

4. Run all\
a. comment the final row `--run_sample` \
b. run sbatch

