import pandas as pd
import os
import argparse
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import json
import utils
from functools import partial
from discriminative_prompts import *
from llms import get_registed_model


def prompt_builder(question, path, mode="zero-shot"):
    if mode == "zero-shot":
        query = ZERO_PROMPT.format(question=question, path=path)
    elif mode == "zero-shot-cot":
        query = ZERO_COT_PROMPT.format(question=question, path=path)
    elif mode == "few-shot":
        query = FEWSHOT_PROMPT.format(question=question, path=path)
    elif mode == "few-shot-cot":
        query = FEWSHOT_COT_PROMPT.format(question=question, path=path)
    else:
        query = ZERO_PROMPT.format(question=question, path=path)
    return query


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, {}
    else:
        with open(path, "r") as f:
            processed_results = {}
            for line in f:
                results = json.loads(line)
                processed_results[results["id"]] = results["acc"]
        fout = open(path, "a")
        return fout, processed_results


def predict(data, args, processed_list, model):
    data_id, row = data
    if data_id in processed_list:
        return None
    question = row["question"]
    ground_truth_paths = row["ground_reasoning_path"]
    result_list = []
    for p in ground_truth_paths:
        path_string = utils.path_to_string(p)
        query = prompt_builder(question, path_string, mode=args.mode)
        query = model.prepare_model_prompt(query)
        response = model.generate_sentence(query)
        prediction = 0
        if "YES" in response.upper():
            prediction = 1
        result_list.append(
            {
                "path": p,
                "prediction": prediction,
                "raw_response": response,
                "raw_input": query,
            }
        )
    avg_result = float(sum([r["prediction"] for r in result_list])) / len(result_list)
    result = {"id": data_id, "acc": avg_result, "details": result_list}
    return result


def main(args, LLM):
    df = pd.read_csv(args.data_path)
    df.rename(columns={" ": "question"}, inplace=True)
    df["ground_reasoning_path"] = df["ground_reasoning_path"].apply(lambda x: eval(x))
    # print(df.columns)

    input_file_name = os.path.basename(args.data_path)
    output_dir = os.path.join(args.output_path, input_file_name, args.model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file_name = f"predictions_{args.mode}{args.postfix}.jsonl"
    output_name = os.path.join(output_dir, output_file_name)

    fout, processed_list = get_output_file(output_name, force=args.force)

    result_list = [value for value in processed_list.values()]

    model = LLM(args)
    print("Prepare pipline for inference...")
    model.prepare_for_inference()

    if args.n > 1:
        with ThreadPool(args.n) as p:
            with tqdm(
                p.imap_unordered(
                    partial(
                        predict, args=args, processed_list=processed_list, model=model
                    ),
                    df.iterrows(),
                ),
                total=len(df),
            ) as phar:
                for r in phar:
                    if r is None:
                        continue
                    fout.write(json.dumps(r) + "\n")
                    result_list.append(r["acc"])
                    if args.debug:
                        for r in r["details"]:
                            print(f"Input: {r['raw_input']}")
                            print(f"Response: {r['raw_response']}")
                            print(f"Prediction: {r['prediction']}")
                    phar.set_postfix(
                        {"ACC": float(sum(result_list)) / len(result_list)}
                    )
    else:
        with tqdm(df.iterrows(), total=len(df)) as phar:
            for data in phar:
                r = predict(data, args, processed_list, model)
                if r is None:
                    continue
                fout.write(json.dumps(r) + "\n")
                result_list.append(r["acc"])
                if args.debug:
                    for r in r["details"]:
                        print(f"Input: {r['raw_input']}")
                        print(f"Response: {r['raw_response']}")
                        print(f"Prediction: {r['prediction']}")
                phar.set_postfix({"ACC": float(sum(result_list)) / len(result_list)})
    fout.close()
    print("Accuracy: ", float(sum(result_list)) / len(result_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="data/Dataset Annotation - cwq_test_res.csv", type=str
    )
    parser.add_argument("--output_path", default="dis_results")
    parser.add_argument("--postfix", default="", type=str)
    parser.add_argument("--n", default=1, type=int, help="number of processes")
    parser.add_argument(
        "--model_name", "-m", type=str, help="model name", default="gpt-3.5-turbo"
    )
    parser.add_argument(
        "--mode",
        default="zero-shot",
        choices=["zero-shot", "zero-shot-cot", "few-shot", "few-shot-cot"],
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    args, _ = parser.parse_known_args()
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()

    main(args, LLM)
