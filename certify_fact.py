import json
import pandas as pd
from tqdm import tqdm
import argparse
import os
import re
from sklearn.metrics import classification_report
import collections
import networkx as nx
from generative_cert.utils.utils import build_graph
from evaluate_results import eval_hit

tqdm.pandas()


class GraphProcess(object):
    def __init__(self, path) -> None:
        self.G = build_graph(path)

    def print(self):
        print(nx.draw_shell(self.G))

    def is_connected_graph(self):
        return nx.is_connected(self.G)

    def find_leaf_nodes(self):
        return [n for n in self.G.nodes() if self.G.degree[n] == 1]

    def test(self):
        check_path = [
            ["Milwaukee Brewers", "sports.sports_team.team_mascot", "Hank"],
            [
                "Miller Park",
                "sports.sports_facility.home_venue_for",
                "Milwaukee Brewers",
            ],
        ]
        print(self.is_connected_graph())
        print(self.find_leaf_nodes())

        check_path = [
            ["Lauren Conrad", "people.person.profession", "Fashion designer"],
            ["Lauren Conrad", "people.place_lived.location", "Los Angeles"],
            [
                "Willie Brown",
                "government.politician.government_positions_held",
                "San Francisco",
            ],
        ]
        print(self.is_connected_graph())
        print(self.find_leaf_nodes())

        check_path = [
            ["Kiribati", "location.country.capital", "South Tarawa"],
            ["Kiribati", "location.location.containedby", "Pacific Ocean"],
            ["Japan", "location.statistical_region.places_exported_to", "Japan"],
        ]
        print(self.is_connected_graph())
        print(self.find_leaf_nodes())


def find_best_triplet(topk_extracted_triplets, prob_thres, ent_thres):
    """
    check if topk extracted triplets is valid
    Valid condition: exist at least 1 extracted triplet
                    whose prob in sententence & entity all are not less than threshold
    """
    twocheck_max = 0
    twocheck_res = None
    sumcheck_max = 0
    sumcheck_res = None
    for triplet_prob in topk_extracted_triplets:
        _, prob, head_score, tail_score = triplet_prob.values()
        head_score = head_score / 100 if isinstance(head_score, int) else head_score
        tail_score = tail_score / 100 if isinstance(tail_score, int) else tail_score
        score = (prob + head_score + tail_score) / 3
        # skip underthreshold triplet
        if not (
            prob >= prob_thres and head_score >= ent_thres and tail_score >= ent_thres
        ):
            if score > sumcheck_max:
                sumcheck_res = triplet_prob
                sumcheck_max = score
        else:
            if score > twocheck_max:
                twocheck_res = triplet_prob
                twocheck_max = score
    return (twocheck_res, twocheck_max) if twocheck_max else (sumcheck_res, 0)


def eval_reasoning_path(path):
    if not path:
        return False
    G = GraphProcess(path)
    return True if G.is_connected_graph() and len(G.G.nodes()) > 2 else False


def certify_prompt_triples(pred_ans, ground_ans, best_triples, prob_thres, ent_thres):
    is_correct_ans = eval_hit(pred_ans, ground_ans)
    res = {
        "p_incorrect_answer": int(not is_correct_ans),
        "p_incorrect_reasoning": 0,
        "p_factual_error": 0,
        "p_coherent_error": 0,
        "p_reasoning_ans_error": 0,
        "p_steps_error": None,
    }
    steps_error = []
    only_tris = []
    order_error = False
    last_tail = None
    # check factual_error
    for i, best_triple in enumerate(best_triples):
        tri, prob, head_score, tail_score = best_triple
        if prob < prob_thres:
            res["p_factual_error"] = 1
            steps_error.append(i + 1)  # due to index start from 0
        else:
            if not last_tail:
                last_tail = [tri[0], tri[2]]
            else:
                if tri[0] in last_tail:
                    last_tail = tri[2]
                elif tri[2] in last_tail:
                    last_tail = tri[0]
                else:
                    order_error = True
        only_tris.append(tri)
    if not only_tris:
        res["p_factual_error"] = 1
        only_tris = [["", "", ""]]

    res["p_steps_error"] = steps_error

    # check coherent error
    res["p_coherent_error"] = (not res["p_factual_error"]) and order_error

    # check answer error
    res["p_reasoning_ans_error"] = (
        (not res["p_factual_error"])
        and (not order_error)
        and (only_tris[-1][-1] not in ground_ans)
    )

    # check reasoning error
    if (
        res["p_factual_error"]
        or res["p_coherent_error"]
        or res["p_reasoning_ans_error"]
    ):  # old version: res['p_incorrect_answer'] or res['p_factual_error']
        res["p_incorrect_reasoning"] = 1
    else:
        res["p_incorrect_reasoning"] = 0  # int(not eval_reasoning_path(only_tris))
    return res


def main(args):
    eval_path = os.path.join(
        args.performance_dir,
        f"{args.dataset}_{args.split}_evaluate_llm_prompting.jsonl",
    )
    retrieval_path = os.path.join(
        args.retrieval_dir,
        args.dataset,
        args.model_name,
        args.split,
        "only_test_extract_triplet_skip_unknown_ent.jsonl",
    )
    topk = args.top_k
    prob_thres = args.prob_thres / 100
    ent_thres = args.ent_thres / 100
    check_valid_list = []  # contain dic: {'id','is_valid','unvalid_step'}

    # check valid in extracted triplet file
    first_line = True
    with open(retrieval_path) as fin:
        for line in tqdm(fin):
            if first_line:
                first_line = False
                continue
            data = json.loads(line)
            topk_extracted_triplets = data[f"{topk}_extracted_triplets"]
            info = {
                "id": data["id"],
                "question": None,
                "prediction": None,  # final answer
                "ground_answer": None,  # groundtruth answer
                "pred_num_steps": None,  # the number of reasoning steps in the answer
                "sent_anwser": None,  # list of sentences in the answer, excluding the final answer
                "prediction_reasoning_path": None,
                "ground_reasoning_path": None,
                "graph": None,
                "raw_graph": None,
                "hit": None,
                "is_valid": True,  # default is True
                "unvalid_step": -1,  # -1 if valid, otherwise is the index of unvalid step
            }
            if not topk_extracted_triplets:
                info = {}
                check_valid_list.append(info)
                continue
            # loop for each step in answer
            sent_anwser = []
            prediction_reasoning_path = []
            for step, sent_k_tri in enumerate(topk_extracted_triplets):
                sent = sent_k_tri["sentence"]
                # skip no meaning sentence
                if "we need" in sent:
                    continue
                triplets = sent_k_tri["triplets"]
                triplet, score = find_best_triplet(triplets, prob_thres, ent_thres)
                # check if extracted triplet is certified
                # certified condition: score>=prob_thres
                if score < prob_thres and info["is_valid"]:
                    info["is_valid"] = False
                    # if uninvalid, point out invalid step
                    info["unvalid_step"] = step
                sent_anwser.append(sent)
                prediction_reasoning_path.append(triplet["triplet"])
            info["sent_anwser"] = sent_anwser
            info["prediction_reasoning_path"] = prediction_reasoning_path
            info["pred_num_steps"] = len(sent_anwser)
            check_valid_list.append(info)

    last_line = len(check_valid_list)
    with open(eval_path) as fin:
        for i, line in enumerate(fin):
            if i == last_line:
                break
            data = json.loads(line)
            if (
                check_valid_list[i]
                and data.get("prediction", None)
                and check_valid_list[i]["id"] == data["id"]
            ):  # skip wrong format
                check_valid_list[i]["question"] = data["question"]
                check_valid_list[i]["prediction"] = data["prediction"]
                check_valid_list[i]["ground_answer"] = data["ground_truth"]
                check_valid_list[i]["hit"] = data["hit"]
                if check_valid_list[i]["is_valid"]:  # only check if certifying fact
                    if not eval_reasoning_path(
                        check_valid_list[i]["prediction_reasoning_path"]
                    ):
                        check_valid_list[i]["is_valid"] = False
                        # if uninvalid, point out invalid step
                        check_valid_list[i]["unvalid_step"] = "reasoning"
                        # check_valid_list[i]['hit'] = 0

    # add ground_reasoning_path & answer sequence
    groundtruthpath_path = (
        f"../data/{args.dataset}/gpt-3.5-turbo/{args.split}/llm_prompt_response.jsonl"
    )
    with open(groundtruthpath_path) as fin:
        first_line = True
        for i, line in enumerate(fin):
            if first_line:
                first_line = False
                continue
            data = json.loads(line)
            i -= 1
            if check_valid_list[i] and check_valid_list[i]["id"] == data["id"]:
                check_valid_list[i]["ground_reasoning_path"] = data[
                    "ground_truth_paths"
                ]
                check_valid_list[i]["sent_anwser"] = [data["reasoning_ans"]] + [
                    check_valid_list[i]["sent_anwser"]
                ]

    # add graph
    test_path = f"../data/{args.dataset}/{args.split}.jsonl"
    graph_dic = {}
    with open(test_path) as fin:
        for line in fin:
            data = json.loads(line)
            graph_dic[data["id"]] = {
                "raw_graph": data["raw_graph"],
                "graph": data["graph"],
            }
    for i in range(len(check_valid_list)):
        if check_valid_list[i]:
            if not graph_dic.get(check_valid_list[i]["id"], None):
                check_valid_list[i] = {}  # cannot find subgraph
            else:
                check_valid_list[i]["raw_graph"] = graph_dic[check_valid_list[i]["id"]][
                    "raw_graph"
                ]
                check_valid_list[i]["graph"] = graph_dic[check_valid_list[i]["id"]][
                    "graph"
                ]

    # check f1
    prediction = [v["is_valid"] for v in check_valid_list if v]
    groundtruth = [v["hit"] for v in check_valid_list if v]
    num_step = [v["pred_num_steps"] for v in check_valid_list if v]
    result = classification_report(
        groundtruth, prediction, target_names=["invalid", "valid"], output_dict=True
    )
    result_str = classification_report(
        groundtruth, prediction, target_names=["invalid", "valid"], output_dict=False
    )

    # print('Hit:', sum(groundtruth)/len(groundtruth))
    # print('Mean of the number of reasoning steps:', sum(num_step)/len(num_step))
    print("The number of reasoning steps:", dict(collections.Counter(num_step)))
    print("Result: \n", result_str)
    with open(
        os.path.join(
            args.out_dir,
            f"{args.dataset}_{args.split}_retrive_fact_certification.jsonl",
        ),
        "w",
    ) as fout:
        fout.write(json.dumps({"args": args.__dict__}) + "\n")
        fout.write(json.dumps(result) + "\n")
        for v in check_valid_list:
            fout.write(json.dumps(v) + "\n")


def replace_sparql(id, old_value, df1):
    value = df1.loc[df1["id"] == id, "groundtruth_paths"]
    if len(value) > 0:
        return value.iloc[0]
    # print('NotFound')
    return old_value


def clear_reasoning_path(paths):
    if not paths:
        return False
    for path in paths:
        for tri in path:
            if re.search("^[mg]\.", tri[0]) or re.search("^[mg]\.", tri[2]):
                return False
    return True


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--performance_dir", type=str, default="experiment_results/evaluate_llm_prompt"
    )
    argparser.add_argument("--retrieval_dir", type=str, default="../data")
    argparser.add_argument(
        "--out_dir", type=str, default="experiment_results/retrive_fact_cert"
    )
    argparser.add_argument("--dataset", type=str, default="cwq")  # cwq, grailqa
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    argparser.add_argument("--top_k", type=int, default=10)
    argparser.add_argument("--prob_thres", type=int, default=70)
    argparser.add_argument("--ent_thres", type=int, default=50)
    args = argparser.parse_args()

    main(args)
    dic_list = []
    # with open(f'LLMReasoningCert/data/{args.dataset}/gpt-3.5-turbo/{args.split}/sparql_ground_truth_paths.jsonl') as fin:
    #     for i, line in enumerate(fin):
    #         data = json.loads(line)
    #         dic_list.append(data)
    # df1 = pd.DataFrame(dic_list)

    dic_list = []
    with open(
        f"experiment_results/retrive_fact_cert/{args.dataset}_{args.split}_retrive_fact_certification.jsonl"
    ) as fin:
        for i, line in enumerate(fin):
            data = json.loads(line)
            dic_list.append(data)
    dic_list = dic_list[2:]
    df = pd.DataFrame(dic_list)
    # df['ground_reasoning_path'] = df.progress_apply(lambda r: replace_sparql(r['id'],r['ground_reasoning_path'], df1), axis=1)
    df.dropna(inplace=True)
    df["sent_anwser"] = df["sent_anwser"].map(lambda x: x[0])
    df["eval"] = df.apply(
        lambda r: f""""hit": {r['hit']}, "is_valid": {r['is_valid']}, "unvalid_step": {r['unvalid_step']}""",
        axis=1,
    )
    df = df[
        [
            "question",
            "ground_answer",
            "sent_anwser",
            "prediction_reasoning_path",
            "ground_reasoning_path",
            "graph",
            "raw_graph",
            "eval",
        ]
    ]

    # df = df[df.ground_reasoning_path.map(clear_reasoning_path)]
    df.to_csv(f"tmp/{args.dataset}_{args.split}_res.csv", index=False)
