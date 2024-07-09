import time
import pandas as pd
import os
import argparse
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import json
import discriminative_cert.utils as utils
from functools import partial
from discriminative_cert.discriminative_prompts import *
from llms import get_registed_model
import wandb
import distutils.util


def prompt_builder(question, answer, path, mode):
    if mode == "zero-shot":
        path_string = utils.path_to_string(path)
        query = ZERO_PROMPT.format(question=question, answer=answer, path=path_string)
    elif mode == "zero-shot-cot":
        path_string = utils.path_to_string(path)
        query = ZERO_COT_PROMPT.format(
            question=question, answer=answer, path=path_string
        )
    elif mode == "few-shot":
        path_string = utils.path_to_string(path)
        query = FEWSHOT_PROMPT.format(
            question=question, answer=answer, path=path_string
        )
    elif mode == "few-shot-cot":
        path_string = utils.path_to_string(path)
        query = FEWSHOT_COT_PROMPT.format(
            question=question, answer=answer, path=path_string
        )
    # elif mode == "neg-few-shot":
    #     path_string = utils.path_to_string(path)
    #     query = NEG_FEWSHOT_PROMPT.format(question=question, path=path_string)
    # elif mode == "neg-few-shot-cot":
    #     path_string = utils.path_to_string(path)
    #     query = NEG_FEWSHOT_COT_PROMPT.format(question=question, path=path_string)
    # elif mode == "neg-reorder-zero-shot":
    #     path_string = utils.reoder_path_to_string(path)
    #     query = ZERO_PROMPT.format(question=question, path=path_string)
    # elif mode == "neg-reorder-few-shot":
    #     path_string = utils.reoder_path_to_string(path)
    #     query = NEG_REORDER_FEWSHOT_PROMPT.format(question=question, path=path_string)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented")
    return query


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, {}
    else:
        fout = open(path, "r+")
        processed_results = {}
        for line in fout:
            try:
                results = json.loads(line)
                processed_results[results["id"]] = results["acc"]
            except:
                print("Error in parsing line: ", line)
                fout.seek(-len(line), 1)
                break
        return fout, processed_results


def predict(data, args, processed_list, model):
    data_id, row = data
    if data_id in processed_list:
        return None
    question = row["question"]
    answer = row["ground_answer"]
    answer_string = " ".join(eval(answer))
    ground_truth_paths = row["ground_reasoning_path"]
    result_list = []
    for p in ground_truth_paths:
        query = prompt_builder(question, answer_string, p, mode=args.mode)
        query = model.prepare_model_prompt(query)
        response = model.generate_sentence(query)
        if response is None:
            continue
        prediction = 0
        if args.eval_neg:
            if "NO" in response.upper() and "YES" not in response.upper():
                prediction = 1
        else:
            if "YES" in response.upper() and "NO" not in response.upper():
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
    result = {
        "id": data_id,
        "ground_answer": answer,
        "question": question,
        "acc": avg_result,
        "details": result_list,
    }
    return result


def main(args, LLM):
    df = pd.read_csv(args.data_path)
    df.rename(columns={" ": "question"}, inplace=True)
    if args.eval_neg:
        df["ground_reasoning_path"] = df["negative_paths"]

    df["ground_reasoning_path"] = df["ground_reasoning_path"].apply(lambda x: eval(x))
    # print(df.columns)

    input_file_name = os.path.basename(args.data_path)
    output_dir = os.path.join(args.output_path, input_file_name, args.model_name)

    while not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(e)
            time.sleep(10)
            pass

    output_file_name = f"predictions_{args.mode}{args.postfix}.jsonl"
    output_name = os.path.join(output_dir, output_file_name)

    fout, processed_list = get_output_file(output_name, force=args.force)

    result_list = [value for value in processed_list.values()]

    model = LLM(args)
    print("Prepare pipline for inference...")
    model.prepare_for_inference()

    if args.wandb:
        wandb.init(
            config=args,
            project="discriminative-cert",
            name=f"{input_file_name}_{args.model_name}_{args.mode}{args.postfix}",
        )

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
    if args.wandb:
        wandb.log({"acc": float(sum(result_list)) / len(result_list)})
    with open(
        os.path.join(output_dir, f"results_{args.mode}{args.postfix}.txt"), "w"
    ) as fout:
        fout.write(f"Accuracy: {float(sum(result_list))/len(result_list)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/cwq_test_res.csv", type=str)
    parser.add_argument("--output_path", default="new_results_w_ans")
    parser.add_argument("--postfix", default="", type=str)
    parser.add_argument("--n", default=1, type=int, help="number of processes")
    parser.add_argument(
        "--model_name", "-m", type=str, help="model name", default="gpt-3.5-turbo"
    )
    parser.add_argument(
        "--mode",
        default="zero-shot",
        choices=[
            "zero-shot",
            "zero-shot-cot",
            "few-shot",
            "few-shot-cot",
            "neg-few-shot",
            "neg-few-shot-cot",
            "neg-reorder-zero-shot",
            "neg-reorder-few-shot",
            "neg-reorder-few-shot-cot",
        ],
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--wandb",
        default=False,
        type=lambda x: bool(distutils.util.strtobool(x)),
        help="enable wandb",
    )
    parser.add_argument(
        "--eval_neg",
        default=False,
        type=lambda x: bool(distutils.util.strtobool(x)),
        help="enable wandb",
    )
    args, _ = parser.parse_known_args()
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()

    main(args, LLM)
