# base on this link: https://github.com/RichardHGL/WSDM2021_NSM/
import json
import sys
import os
import time
import pandas as pd
from datasets import Dataset, load_dataset
from LLMReasonCert.extract_subgraph.deal_cvt import load_cvt, is_cvt


def get_domain_from_dataset(dat_path):
    with open(dat_path) as fin:
        data = json.load(fin)
        data = data["Questions"]
        filter_domain = set()
        for ques in data:
            parses = ques["Parses"]
            for parse in parses:
                rel_str = parse["InferentialChain"].split("..")
                filter_domain.update(rel_str)
    return list(filter_domain)


def manual_filter_rel(filter_domain):
    filter_set = set(filter_domain)
    input = "LLMReasoningCert/data/data/fb_en.txt"
    output = "LLMReasoningCert/data/data/manual_fb_filter.txt"
    f_in = open(input)
    f_out = open(output, "w")
    num_line = 0
    num_reserve = 0
    for line in f_in:
        splitline = line.strip().split("\t")
        num_line += 1
        if len(splitline) < 3:
            continue
        rel = splitline[1]
        flag = False
        for domain in filter_set:
            if domain in rel:
                flag = True
                break
        if flag:
            continue
        f_out.write(line)
        num_reserve += 1
        if num_line % 1000000 == 0:
            print("Checked: ", num_line, "lines, Reserver:", num_reserve)
    f_in.close()
    f_out.close()
    print("Total: ", num_line, "lines, Reserver:", num_reserve)


def is_ent(tp_str):
    if len(tp_str) < 3:
        return False
    if tp_str.startswith("m.") or tp_str.startswith("g."):
        return True
    return False


def fetch_triple_1hop(kb_file, seed_set, cvt_nodes, cvt_hop=True):
    cvt_set = set()
    num_tot = 0
    num_res = 0
    subgraph = []
    f = open(kb_file)
    for line in f:
        spline = line.strip().split("\t")
        num_tot += 1
        if spline[0] in seed_set:
            # Only subject is enough.
            if cvt_hop and spline[2] not in seed_set and is_cvt(spline[2], cvt_nodes):
                cvt_set.add(spline[2])
            subgraph.append(spline)
            num_res += 1
        if num_tot % 1000000 == 0:
            print("seed-hop", num_tot, num_res)

    num_tot = 0
    num_res = 0
    if cvt_hop:
        cvt_set = cvt_set - seed_set
        with open(kb_file) as f:
            for line in f:
                num_tot += 1
                spline = line.strip().split("\t")
                if spline[0] in cvt_set:
                    subgraph.append(spline)
                    num_res += 1
                if num_tot % 1000000 == 0:
                    print("seed-hop", num_tot, num_res)
    return subgraph


def filter_ent_from_triple(subgraph):
    ent_set = set()
    for line in subgraph:
        if is_ent(line[0]):
            ent_set.add(line[0])
        if is_ent(line[2]):
            ent_set.add(line[2])
    return ent_set


def get_n_hop_supgraph(seed_set, n_hop, cvt_nodes):
    st = time.time()
    kb_file = "LLMReasoningCert/data/data/manual_fb_filter.txt"

    for i_hop in range(1, n_hop + 1):
        subgraph = fetch_triple_1hop(
            kb_file=kb_file, seed_set=seed_set, cvt_nodes=cvt_nodes, cvt_hop=True
        )
        print(f"\tHop {i_hop}", time.time() - st)
        st = time.time()

        if i_hop < n_hop:
            hop1_ent = filter_ent_from_triple(subgraph=subgraph)
            print("\tFetch ent from Hop 1", time.time() - st)
            st = time.time()
            seed_set = hop1_ent

    print("\tDone.", time.time() - st)
    return subgraph


def freebase_supgraph(data_folder, n_hop):
    def add_subgraph_to_parse(parse, cvt_nodes, n_hop):
        for parse_i in range(len(parse)):
            seed_set = set(
                [parse[parse_i]["TopicEntityMid"]]
                + [ans["AnswersMid"] for ans in parse[parse_i]["Answers"]]
            )
            parse[parse_i]["subgraph"] = get_n_hop_supgraph(seed_set, n_hop, cvt_nodes)
        return parse

    data_file = [
        "FreebaseQA-dev.json",
        "FreebaseQA-eval.json",
        "ComplexWebQuestions_train.json",
    ]
    input_dir = "raw"

    st = time.time()
    cvt_nodes = load_cvt()
    print("Loaded CVT", time.time() - st)

    for file in data_file:
        # print(f'Handled {count}/{len_data} in {file}')
        print(f"Handling {file}")
        output_file = os.path.join(data_folder, file)
        input_file = os.path.join(data_folder, input_dir, file)
        with open(input_file) as f_in:
            data = json.load(f_in)
            data = Dataset.from_pandas(pd.DataFrame(data=data["Questions"]))
            data = data.map(
                input_columns=["Parses"],
                remove_columns=["Parses"],
                function=lambda x: {
                    "Parses": add_subgraph_to_parse(x, cvt_nodes, n_hop)
                },
                num_proc=8,
            )
        data.to_json(output_file)


def grailqa_supgraph(data_dir, n_hop):
    def add_subgraph_to_parse(row, cvt_nodes, n_hop):
        seed_set = set(
            [row["graph_query"]["id"][-1]] + row["answer"]["answer_argument"]
        )
        return get_n_hop_supgraph(seed_set, n_hop, cvt_nodes)

    st = time.time()
    cvt_nodes = load_cvt()
    print("Loaded CVT", time.time() - st)

    data_dic = load_dataset("grail_qa")
    for split in data_dic:
        output_file = os.path.join(data_dir, f"{split}.json")
        data = data_dic[split]
        data = data.map(
            lambda x: {"subgraph": add_subgraph_to_parse(x, cvt_nodes, n_hop)},
            num_proc=8,
        )
    data.to_json(output_file)


if __name__ == "__main__":
    # step0: filter freebase: not triplet
    # dat_path = 'LLMReasoningCert/data/FreebaseQA/raw/'
    # filter_domain = get_domain_from_dataset(dat_path)
    # print(filter_domain[:2])
    # manual_filter_rel([])

    # step1: get subgraph
    # freebase
    # freebase_supgraph(data_folder='LLMReasoningCert/data/FreebaseQA', dat_name='freebase', n_hop=2)
    grailqa_supgraph(
        data_folder="LLMReasoningCert/envs/huggingface",
        dat_name="grail_qa",
        n_hop=4,
    )
