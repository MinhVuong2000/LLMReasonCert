import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
import torch
from datasets import load_dataset, Dataset
from generative_cert.prompts import *
from llms import get_registed_model


def prompt_builder(question, hint, mode="zero-shot"):
    if mode == "fewshot-cot-hint":
        query = FEWSHOT_COT_HINT.format(question=question)
    elif mode == "fewshot-cot-hint-ground":
        query = FEWSHOT_COT_HINT_GROUND.format(question=question, hint=hint)
    else:
        query = FEWSHOT_COT_ONLY.format(question=question)
    return query


def load_data(path, dataset):
    in_path = path.format(dataset)
    with open(in_path, "r") as fin:
        results = json.load(fin)
        # ignore_1hop:
        results = results["min_2hop"] + results["min_multihop"]
        dat = pd.DataFrame(results)[["id", "question", "ground_truth_paths"]]
    return dat


def write_results(out_dir, file_name):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, file_name)
    fout = open(out_path, "w")
    return fout


def predict(data, args, model):
    data_id, row = data
    question = row["question"]
    hint = " -> ".join([tri[1] for tri in row["ground_truth_paths"][0]])

    query = prompt_builder(question, hint, args.mode)
    query = model.prepare_model_prompt(query)
    response = model.generate_sentence(query)
    row["reasoning_ans"] = response
    return row


def main(args, LLM):
    model = LLM(args)
    print("Prepare pipeline for inference...")
    model.prepare_for_inference()
    if args.dataset == "all":
        dataset = ["cwq", "FreebaseQA"]
    else:
        dataset = [args.dataset]
    for d in dataset:
        out_dir, file_name = (
            os.path.join(args.out_dir, d, args.model_name, args.exp_name),
            "llm_prompt_response.jsonl",
        )
        fout = write_results(out_dir, file_name)
        ques_dat = load_data(args.in_dir, d)
        if args.run_sample:
            ques_dat = ques_dat.iloc[:3]
        with tqdm(ques_dat.iterrows(), total=len(ques_dat)) as phar:
            for data in phar:
                r = predict(data, args, model)
                fout.write(json.dumps(r.to_dict()) + "\n")
        fout.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", help="LLM", type=str, required=True)
    argparser.add_argument(
        "--mode",
        help="LLM",
        type=str,
        choices=["fewshot-cot-only", "fewshot-cot-hint", "fewshot-cot-hint-ground"],
        default="fewshot-cot-only",
        required=True,
    )
    argparser.add_argument(
        "--dataset",
        help="dataset name",
        type=str,
        choices=["cwq", "FreebaseQA", "grail_qa", "all"],
        required=True,
    )
    argparser.add_argument(
        "--in_dir",
        help="directory containing question data",
        type=str,
        default="LLMReasoningCert/data/{}/gpt-3.5-turbo/test/splitted_ground_truth_paths.json",
    )
    argparser.add_argument(
        "--out_dir",
        help="directory containing answer output data",
        type=str,
        default="LLMReasoningCert/data",
    )
    argparser.add_argument(
        "--run_sample", help="only run 3 samples", action="store_true"
    )
    argparser.add_argument("--exp_name", type=str, default="", help="Experiment name")

    args, _ = argparser.parse_known_args()
    LLM = get_registed_model(args.model_name)
    LLM.add_args(argparser)
    args = argparser.parse_args()

    main(args, LLM)
