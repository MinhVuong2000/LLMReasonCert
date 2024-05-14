from generative_cert.utils.utils import build_graph, get_edge_data
import random
import pickle
import pandas as pd
import os
import json
from tqdm import tqdm

tqdm.pandas()


def create_misguide_path(G, start_node, end_nodes, num_hop):
    def create_random_path(G, start_node, num_hop):
        random_path = []
        current_node = start_node
        for _ in range(num_hop):
            neighbors = list(G.neighbors(current_node))
            if neighbors:
                next_node = random.choice(neighbors)
                rel = get_edge_data(G, current_node, next_node)
                random_path.append([current_node, rel, next_node])
                current_node = next_node
            else:
                break
        return random_path

    misguide_path = None
    for _ in range(20):
        if (not misguide_path) or (misguide_path[-1][-1] in end_nodes):
            misguide_path = create_random_path(G, start_node, num_hop)
    if (misguide_path[-1][-1] in end_nodes) or len(misguide_path) < 2:
        raise ValueError("cannot find a misguide path")
    return misguide_path


def misguide_path_from_ref_paths(G, ground_truth_paths):
    # get question entity and answer entities from ground_truth_paths
    ques_ent, ans_ents = None, []
    if isinstance(ground_truth_paths[0][0], str):
        ques_ent = ground_truth_paths[0][0]
        ans_ents = [ground_truth_paths[-1][-1]]
    else:
        ques_ent = ground_truth_paths[0][0][0]
        ans_ents = [path[-1][-1] for path in ground_truth_paths]

    # create misguide path
    num_hop = random.choice([2, 3])
    misguide_path = create_misguide_path(G, ques_ent, ans_ents, num_hop)
    for _ in range(50):
        if [
            p
            for p in ground_truth_paths
            if "".join(["".join(t) for t in p])
            in "".join(["".join(t) for t in misguide_path])
        ]:
            misguide_path = create_misguide_path(G, ques_ent, ans_ents, num_hop)
        else:
            break
    if [
        p
        for p in ground_truth_paths
        if "".join(["".join(t) for t in p])
        in "".join(["".join(t) for t in misguide_path])
    ]:
        raise ValueError("misguide_path contains ground_truth_paths")
    return misguide_path


def create_misguide_path_for_dat(dataset, out_path):
    # load triplets
    triplets_path = (
        f"LLMReasoningCert/data/db_extract/{dataset}/only_test_set/origin/triplets.pkl"
    )
    with open(triplets_path, "rb") as f:
        triplets = pickle.load(f)
    # create graph for whole triplets
    G = build_graph(triplets)

    # load data from dataset
    in_path = "LLMReasoningCert/data/{}/gpt-3.5-turbo/test/splitted_ground_truth_paths.json".format(
        dataset
    )
    with open(in_path, "r") as fin:
        results = json.load(fin)
    # ignore_1hop:
    results = results["min_2hop"] + results["min_multihop"]
    dat = pd.DataFrame(results)[["id", "question", "ground_truth_paths"]]

    # get misguide paths
    dat["misguide_path"] = dat["ground_truth_paths"].progress_apply(
        lambda p: misguide_path_from_ref_paths(G, p)
    )

    # save
    dat.to_json(out_path, orient="records")
    print(
        dat.apply(
            lambda r: r["misguide_path"][-1][-1]
            in [p[-1][-1] for p in r["ground_truth_paths"]],
            axis=1,
        ).sum()
        / len(dat)
    )


if __name__ == "__main__":
    dataset = "grail_qa"  # cwq
    out_path = f"LLMReasoningCert/data/{dataset}/misguide_path.jsonl"
    create_misguide_path_for_dat(dataset, out_path)

    df = pd.read_json(out_path)
    func = lambda ground_truth_paths, misguide_path: len(
        [
            p
            for p in ground_truth_paths
            if "".join(["".join(t) for t in p])
            in "".join(["".join(t) for t in misguide_path])
        ]
    )
    print(
        df.apply(
            lambda r: func(r["ground_truth_paths"], r["misguide_path"]), axis=1
        ).sum()
    )
