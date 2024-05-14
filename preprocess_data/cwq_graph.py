import json
from preprocess_data.sparql import SparQL
from tqdm import tqdm
from datasets import load_dataset


def get_topic_entities(dataset="cwq", split="test"):
    path = f"LLMReasoningCert/data/{dataset}/old_data/{split}.jsonl"
    res = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            res[data["id"]] = {
                "q_entity": data["q_entity"],
                "answer": data["answer"],
                "a_entity": data["a_entity"],
            }
    return res


def get_data(dataset="cwq", split="ComplexWebQuestions_test", short_split="test"):
    old_data = get_topic_entities(dataset, short_split)
    path = f"LLMReasoningCert/data/{dataset}/raw/{split}.json"
    out_path = f"LLMReasoningCert/data/{dataset}/{short_split}.jsonl"
    with open(path) as f:
        data = json.load(f)
    sparql = SparQL("http://localhost:3001/sparql")
    with open(out_path, "w") as f:
        out_count = 0
        for sample in tqdm(data):
            id = sample["ID"]
            expand_info = old_data.get(id, None)
            if not expand_info:
                continue
            try:
                rng_path, processed_rng_path = sparql.query_reasoning_path(
                    sample["sparql"]
                )
            except Exception as e:
                print(e)
                continue
            dic = {
                "id": id,
                "question": sample["question"],
                "q_entity": expand_info["q_entity"],
                "a_entity": expand_info["a_entity"],
                "answer": expand_info["answer"],
                "graph": processed_rng_path,
                "raw_graph": rng_path,
            }
            f.write(json.dumps(dic) + "\n")
            out_count += 1
    print(f"CWQ: Wrote: {out_count}/{len(data)} samples")


if __name__ == "__main__":
    get_data(dataset="cwq", split="ComplexWebQuestions_test", short_split="test")
