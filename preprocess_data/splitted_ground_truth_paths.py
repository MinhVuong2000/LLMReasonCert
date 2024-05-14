import json
import os
from itertools import chain
from generative_cert.utils.utils import is_unknown_ent


def count_min_hop_path(ground_truth_paths):
    min_hop = min(map(len, ground_truth_paths))
    return min_hop


def rearrange_paths(inp_f, out_f):
    with open(out_f, "w") as fout:
        dic = {"min_1hop": [], "min_2hop": [], "min_multihop": []}
        with open(inp_f, "r") as fin:
            for line in fin:
                res = json.loads(line)
                if count_min_hop_path(res["ground_truth_paths"]) < 2:
                    dic["min_1hop"].append(res)
                elif count_min_hop_path(res["ground_truth_paths"]) < 3:
                    dic["min_2hop"].append(res)
                else:
                    dic["min_multihop"].append(res)
        fout.write(json.dumps(dic, indent=4))


def count_hop_cate(inp_f, out_f):
    with open(out_f, "w") as fout:
        dic = {}
        with open(inp_f, "r") as fin:
            for line in fin:
                res = json.loads(line)
                if count_min_hop_path(res["ground_truth_paths"]) < 2:
                    dic["min_1hop"].append(res)
                elif count_min_hop_path(res["ground_truth_paths"]) < 3:
                    dic["min_2hop"].append(res)
                else:
                    dic["min_multihop"].append(res)
        fout.write(json.dumps(dic, indent=4))


if __name__ == "__main__":
    d_l = ["grail_qa", "cwq"]
    split_l = ["test"]

    data_path = "LLMReasoningCert/data/"
    ground_truth_paths_file = os.path.join(
        data_path, "{}/gpt-3.5-turbo/{}/ground_truth_paths.jsonl"
    )
    ground_truth_multi_paths_file = os.path.join(
        data_path, "{}/gpt-3.5-turbo/{}/splitted_ground_truth_paths.json"
    )

    dic = {}
    for d in d_l:
        for split in split_l:
            print("Handling data {} and split {}: ".format(d, split))
            rearrange_paths(
                ground_truth_paths_file.format(d, split),
                ground_truth_multi_paths_file.format(d, split),
            )
            with open(ground_truth_multi_paths_file.format(d, split), "r") as fin:
                res = json.load(fin)
                dic[f"{d}_{split}"] = {
                    "1hop": len(res["min_1hop"]),
                    "2hop": len(res["min_2hop"]),
                    "multihop": len(res["min_multihop"]),
                }
    with open(os.path.join(data_path, "stat.json"), "w") as fout:
        fout.write(json.dumps(dic, indent=2))
