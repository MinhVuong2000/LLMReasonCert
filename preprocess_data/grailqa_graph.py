import re
import os
import json
import warnings
from tqdm import tqdm
from datasets import load_dataset
from preprocess_data.sparql import SparQL
from generative_cert.utils.utils import is_unknown_ent

warnings.filterwarnings("ignore")
num_proc = 16

sparql = SparQL("http://localhost:3001/sparql")


def get_ent_in_graphquery(dic):
    ids = dic["nodes"]["id"]
    ids = [e for e in ids if (is_unknown_ent(e) or "^^http:" in e)]
    ids = [e.split("^^")[0] for e in ids]
    res = [sparql.SQL_entity2name(e) for e in ids if e != "0"]
    if any([is_unknown_ent(e) for e in res]):
        raise ValueError()
    return res


def remove_for_searching_subgraph(sparql_txt):
    sparql_txt = sparql_txt.replace("\n", "").replace("}", " }")
    sparql_txt = re.sub(r" ?VALUES \?\w\d { ((:[mg]\.\w+)|\".+\>) +}", "", sparql_txt)
    return sparql_txt


def subgraph_comparative_ques(sparql_txt):
    #  replace comparative entities with not in KG
    comparative_ent = re.search(r"\?x\d [<>]=? \"", sparql_txt).group()[
        :3
    ]  # return ?x\d
    sparql_txt1 = sparql_txt.replace(
        "SELECT (?x0 AS ?value) WHERE { SELECT DISTINCT ?x0",
        "SELECT (?x0 AS ?value) WHERE { SELECT DISTINCT ?x0".replace(
            "?x0", comparative_ent
        ),
    )
    added_q_entities = sparql.query(sparql_txt1, variable="value")[1]
    # find subgraph
    sparql_txt = remove_for_searching_subgraph(sparql_txt)
    raw_subgraph, subgraph = sparql.query_reasoning_path(sparql_txt)
    return raw_subgraph, subgraph, added_q_entities, []


def subgraph_count_ques(sparql_txt):
    # search ans ents due to the final answer is COUNT=number
    added_a_entities = sparql.query(sparql_txt.replace("COUNT(?x0)", "?x0"))[1]
    # find subgraph
    sparql_txt = remove_for_searching_subgraph(sparql_txt)
    raw_subgraph, subgraph = sparql.query_reasoning_path(sparql_txt)
    return raw_subgraph, subgraph, [], added_a_entities


def subgraph_superlative_ques(sparql_txt):
    sparql_txt = remove_for_searching_subgraph(sparql_txt)
    sparql_txt = re.sub(r"WHERE { \?y\d .+} ?\?x", "WHERE { ?x", sparql_txt)
    sparql_txt = re.sub(r" FILTER ( \?y.+ \?y\d )", "", sparql_txt)
    raw_subgraph, subgraph = sparql.query_reasoning_path(sparql_txt)
    return raw_subgraph, subgraph, [], []


def subgraph_none_ques(sparql_txt):
    sparql_txt = remove_for_searching_subgraph(sparql_txt)
    raw_subgraph, subgraph = sparql.query_reasoning_path(sparql_txt)
    return raw_subgraph, subgraph, [], []


def find_subgraph(row):
    sparql_txt = row["sparql_query"]
    # find subgraph
    try:
        if row["function"] == "none":
            raw_subgraph, subgraph, added_q_entities, added_a_entities = (
                subgraph_none_ques(sparql_txt)
            )
        elif row["function"] == "count":
            raw_subgraph, subgraph, added_q_entities, added_a_entities = (
                subgraph_count_ques(sparql_txt)
            )
        elif row["function"] in ["argmax", "argmin"]:
            raw_subgraph, subgraph, added_q_entities, added_a_entities = (
                subgraph_superlative_ques(sparql_txt)
            )
        else:  # row['function'] in [>=, <=, >, <]
            raw_subgraph, subgraph, added_q_entities, added_a_entities = (
                subgraph_comparative_ques(sparql_txt)
            )
    except Exception as e:
        # print(e)
        with open("temp.txt", "a") as f:
            for k, v in row.items():
                f.write(f"{k}:{v}" + "\n")
            f.write("\n")
        # return None, None
        raise ValueError()

    return raw_subgraph, subgraph, added_q_entities, added_a_entities


def get_info(row, fout, processed_ids):
    id = row["qid"]
    if id in processed_ids:
        return {"id": id, "status": "processed"}
    ques = row["question"]
    ans_ent = (
        row["answer"]["entity_name"]
        if row["answer"]["entity_name"] != [""]
        else row["answer"]["answer_argument"]
    )  # list of answer entities
    topic_ent = get_ent_in_graphquery(row["graph_query"])  # list of topic entities
    if not topic_ent:
        return {"id": id, "status": "none"}
    else:
        raw_subgraph, subgraph, added_q_entities, added_a_entities = find_subgraph(row)
    if not subgraph:
        return {"id": id, "status": "none"}
    topic_ent += added_q_entities
    ans_ent += added_a_entities
    res = {
        "id": id,
        "question": ques,
        "q_entity": topic_ent,
        "a_entity": ans_ent,
        "answer": ans_ent,
        "graph": subgraph,
        "raw_graph": raw_subgraph,
        "function": row["function"],
    }
    fout.write(json.dumps(res) + "\n")
    return {"id": id, "status": "sucessed"}


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, {}
    else:
        with open(path, "r") as f:
            processed_ids = []
            for line in f:
                results = json.loads(line)
                processed_ids.append(results["id"])
        fout = open(path, "a")
        # print('processed_ids', processed_ids)
        return fout, processed_ids


def get_multihop(dataset="grail_qa", split="validation"):
    dat = load_dataset(dataset, split=split)
    # ignore_1hop:
    dat = dat.filter(lambda r: r["num_edge"] > 1)
    return dat


def get_data(dataset="grail_qa", split="validation"):
    if split == "validation":
        convert_split = "test"
    out_path = f"LLMReasoningCert/data/{dataset}/{convert_split}.jsonl"
    print(out_path)
    fout, processed_ids = get_output_file(out_path)
    dat = get_multihop(dataset, split)
    dat = dat.map(
        lambda row: get_info(row, fout, processed_ids),
        num_proc=num_proc,
        remove_columns=dat.column_names,
    )
    fout.close()
    print(dat.to_pandas().status.value_counts())
    print(f"GrailQA: Done")


if __name__ == "__main__":
    get_data()
