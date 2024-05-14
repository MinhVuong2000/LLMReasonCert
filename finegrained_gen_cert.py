import os
import json
import argparse
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
from datasets import load_dataset
from minineedle import needle


def prec_eval(row, prob_thres=0.7, ent_thres=0.5):
    # init precision value
    max_match = 0.0
    min_edit = 1.0
    p_incorrect_reasoning = 0

    # get triples
    ref_path = row["ground_truth_paths"]
    extracted_triplets = row["10_extracted_triplets"]
    pred_path = [
        [
            v["best_triple"]["triplet"]
            if (
                v["best_triple"]["prob"] >= prob_thres
                and v["best_triple"]["head_score"] >= ent_thres
                and v["best_triple"]["tail_score"] >= ent_thres
            )
            else ["h", "r", "t"]
            for v in res
        ]
        for res in extracted_triplets
    ]
    # sort and linearize to make sure the order in path
    ref_path = [[" ".join(sorted(v)) for v in res] for res in ref_path]
    pred_path = [[" ".join(sorted(v)) for v in res] for res in pred_path]

    for pred in pred_path:  # check both self-consistency & a seq
        for ref in ref_path:  # check each reference path
            alignment = needle.NeedlemanWunsch(pred, ref)
            alignment.align()
            score = alignment._score
            match = int(
                round(alignment._identity / 100 * len(alignment._alseq1), 0)
            ) / len(ref)  # maybe exist redundant
            edit = 1 - alignment._identity / 100
            if match > max_match:
                max_match = match
                min_edit = edit
            elif match == max_match:
                if edit < min_edit:
                    min_edit = edit

    p_incorrect_reasoning = 0 if match == 1 else 1
    p_incorrect_answer = row["short_eval"]["p_incorrect_answer"]
    id = row["id"]
    return {
        "id": id,
        "p_incorrect_answer": p_incorrect_answer,
        "p_incorrect_reasoning": p_incorrect_reasoning,
        "match_rate": match,
        "edit_rate": min_edit,
    }


def calc_and_save_result_metrics(out_dir, dat):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    dat.to_json(os.path.join(out_dir, "data.jsonl"))

    ans_acc = (
        round(len(dat.filter(lambda x: x["p_incorrect_answer"] == 0)) / len(dat), 4)
        * 100
    )
    reasoning_acc = (
        round(len(dat.filter(lambda x: x["p_incorrect_reasoning"] == 0)) / len(dat), 4)
        * 100
    )
    # confusion matrix
    classes = [1, 0]
    dat = dat.to_pandas()
    cfm = confusion_matrix(dat["p_incorrect_answer"], dat["p_incorrect_reasoning"])
    tn, fp, fn, tp = cfm.ravel()
    ## dic
    cfm_dic = {
        "IA_UR": str(tp),  # incorrect answer + uncertified reasoning
        "CA_UR": str(fp),  # correct answer + uncertified reasoning
        "IA_CR": str(fn),  # incorrect answer + certified reasoning
        "CA_CR": str(tn),  # correct answer + certified reasoning
    }
    ## image
    cfm_df = pd.DataFrame(
        cfm,
        index=["correct answer", "incorrect answer"],
        columns=["certified reasoning", "uncertified reasoning"],
    )
    sn.heatmap(cfm_df, annot=True, fmt="d").figure.savefig(
        os.path.join(out_dir, "confusion_matrix.png")
    )

    # write
    with open(os.path.join(out_dir, "metric.txt"), "w") as fout:
        fout.write(f"answer_accuracy {ans_acc}%" + "\n")
        fout.write(f"reasoning_accuracy {reasoning_acc}%" + "\n")
        fout.write(json.dumps(cfm_dic))


def main(args, prob_thres, ent_thres):
    # load data
    path = args.dat_path
    dat = load_dataset("json", data_files=path)["train"]
    # calculate precision
    dat = dat.map(
        lambda x: prec_eval(x, prob_thres, ent_thres),
        batched=False,
        num_proc=16,
        remove_columns=dat.column_names,
    )
    # save
    out_dir = os.path.join(os.path.dirname(path), "groundtruth")
    calc_and_save_result_metrics(out_dir, dat)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dat_path",
        type=str,
        required=True,
        help="LLMReasoningCert/data/cwq/gpt-3.5-turbo/gpt-3.5-turbo-fewshot-cot-only-cwq-temp-0.7-p-0.9-consistency-1/llm_prompt_response.jsonl",
    )
    args = argparser.parse_args()

    prob_thres = 0.7
    ent_thres = 0.5
    main(args, prob_thres, ent_thres)
