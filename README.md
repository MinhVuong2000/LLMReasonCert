Direct Evaluation of CoT in Multi-hop Reasoning with Knowledge Graphs
---
Official Implementation of ["Direct Evaluation of Chain-of-Thought in Multi-hop Reasoning with Knowledge Graphs"](https://arxiv.org/abs/2402.11199).

Has been accepted at [ACL2024](https://2024.aclweb.org/) Findings.

<img src="./figures/framework.png" width = "800" />

Aiming evaluate not only final answers but also intermediate steps in the CoT reasoning capabilities of LLMs in multi-hop question answering, the paper proposed 2 evaluation modules: 
1. **Discriminative**: assess LLMs' knowledge of reasoning
2. **Generative**: assess the accuracy of the generated CoT by utilizing knowledge graphs (KGs).

In addition, we do ablation studies to evaluate the fine-grain CoT generation to calculate edit-distance & reasoning errors.

## Requirements
```sh
conda create --name llm-reasoning-cert python=3.8
conda activate llm-reasoning-cert
```
```sh
pip install -r requirements.txt
```

## Datasets
The paper uses 2 datasets: [CWQ](https://allenai.org/data/complexwebquestions) and [GrailQA](https://huggingface.co/datasets/grail_qa) as initiate datasets for experiments.

Then, extract subgraph and ground-truth reasoning path based on SPARQL.

Final datasets used for the paper are uploaded into HuggingFace: (Note: update later)
1. [CWQ-Subgraph-Eval]()
2. [GrailQA-Subgraph-Eval]()

### Preprocess for each dataset: 
Aim: create subgraphs for querying ground-truth reasoning path & creating VectorDB
#### Create subgraphs
Code at `./preprocess_data`
1. Create **subgraph** from the **raw-subgraph** via the detail implementation in [preprocess's readme](./preprocess_data/readme.md) 
3. Get **groundtruth reasoning path** via the **subgraph**, `answer entities` and `topic entities`
```bash
python ./preprocess_data/ground_truth_paths.py
```
4. Rearrange questions according to the number of edge of **groundtruth reasoning path**
```bash
python ./preprocess_data/splitted_ground_truth_paths.py
```
We only use questions >=2 hops in the corresponding reasoning path.

#### Create VectorDB
`FAISS` & `sentence-transformers/all-mpnet-base-v2` are used to create VectorDB before retrieving 
```bash
DATASET='cwq' # 'grail_qa
sbatch scripts/gen-cert/extract_triplet.sh $DATASET
```
you can setup addition arguments: 
- embed_model_name. Default is `sentence-transformers/all-mpnet-base-v2`
- top_k. Default is `10`
- device. Default is `cpu`

However, remember re-setup them in `./generative-cert.py#L228`

## How to run
Set your OpenAI api key & Huggingface key (if needed) in `.env` (check file `.env.example` as the example).

### Discriminative Mode
```bash
    sh scripts/submit_discriminative_cert.sh
```

### Generative Mode
#### Stage1: LLM prompting for structured answer
1. ChatGPT
```bash
sh scripts/gen-cert/llm_prompting.sh
```
2. HF models: Llama2 7B/13B/70B chat-hf, Mistral-7B-Instruct-v0.1, Qwen-14B-Chat, Vicuna-33b-v1.3
```bash
sh generative_cert/scripts/fitcluster/script.sh
```

#### Stage 2 & 3: Retrieval & Evaluation
1. Main result
```bash
sh scripts/gen-cert/job_eval_llm.sh
```
2. The fine-grained generative evaluation: edit-distance score
```bash
sh scripts/gen-cert/job_eval_llm_finegrained.sh
python finegrained_analysis.py
```
3. Run the analysis for reasoning errors
```bash
python finegrained_analysis.py
```

## Results
<img src="./figures/discriminative_result.png" width = "600" />
<img src="./figures/generative_result.png" width = "900" />

---
## Citation
If you find this paper or the repo useful for your work, please consider citing the paper
```
@misc{nguyen2024direct,
    title={Direct Evaluation of Chain-of-Thought in Multi-hop Reasoning with Knowledge Graphs},
    author={Minh-Vuong Nguyen and Linhao Luo and Fatemeh Shiri and Dinh Phung and Yuan-Fang Li and Thuy-Trang Vu and Gholamreza Haffari},
    year={2024},
    eprint={2402.11199},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
