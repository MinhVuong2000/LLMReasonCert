import re
import json
import os


def check_type_question(dir, filename):
    data = []
    for path in filename:
        inp_path = os.path.join(dir, path)
        sample = json.load(inp_path)
        samples = sample["min_2hop"] + sample["min_multihop"]
        for ques in sample:
            data.append(ques["question"])

    res_dic = {
        "atleast_atmost_which": [],
        "atleast_atmost_count": [],
        "more_less_which": [],
        "more_less_count": [],
        "bool": [],
        "min_max": [],
        "count": [],
        "logical": [],
        "unknown": [],
    }
    for ques in data:
        ques = ques.lower()
        if " more " in ques or " less " in ques:
            if "which" in ques:
                res_dic["more_less_which"].append(ques)
            elif "how many" in ques:
                res_dic["more_less_count"].append(ques)
            else:
                res_dic["unknown"].append(ques)
        elif " at least " in ques or " at most " in ques:
            if "which" in ques:
                res_dic["atleast_atmost_which"].append(ques)
            elif "how many" in ques:
                res_dic["atleast_atmost_count"].append(ques)
            else:
                res_dic["unknown"].append(ques)
        elif re.search("^(does |do |is |are )", ques):
            res_dic["bool"].append(ques)
        elif re.search("minimum|maximum|largest|smallest", ques):
            res_dic["min_max"].append(ques)
        elif "how many" in ques:
            res_dic["count"].append(ques)
        elif re.search("what|which", ques) in ques and re.search(
            " and | or | not ", ques
        ):
            res_dic["logical"].append(ques)
        else:
            res_dic["unknown"].append(ques)
    with open(os.path.join(dir, "check_type.json"), "w") as fout:
        json.dump(res_dic, fout, indent=4)


def count_hop_grailQA(
    cache_dir="LLMReasoningCert/data",
):
    data = load_dataset("grail_qa", cache_dir=cache_dir)
    dic = {}
    for split in data:
        if split == "test":
            continue
        dic[split] = {}
        for v in data[split]["num_edge"]:
            if v not in dic[split]:
                dic[split][v] = 1
            else:
                dic[split][v] += 1
    return dic


def count_hop_fbQA(dat_path):
    with open(dat_path) as fin:
        data = json.load(fin)
        data = data["Questions"]
        filter_domain = set()
        hop_dic = {}
        for ques in data:
            min_num_hop = min(
                [len(parse["InferentialChain"].split("..")) for parse in ques["Parses"]]
            )
            if min_num_hop not in hop_dic:
                hop_dic[min_num_hop] = [ques["Question-ID"]]
            else:
                hop_dic[min_num_hop].append(ques["Question-ID"])
    return {k: len(v) for k, v in hop_dic.items()}


if __name__ == "__main__":
    dir = "LLMReasoningCert/data/cwq/"
    check_type_question(
        dir,
        filename=[
            "gpt-3.5-turbo/train/splitted_ground_truth_paths.json",
            "gpt-3.5-turbo/test/splitted_ground_truth_paths.json",
            "gpt-3.5-turbo/dev/splitted_ground_truth_paths.json",
        ],
    )
