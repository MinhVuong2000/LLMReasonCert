import generative_cert.utils.utils as utils
import os
import argparse
from datasets import load_dataset
import json
from tqdm import tqdm
import multiprocessing as mp
from extract_subgraph.graph_loader import GraphProcess
from functools import partial
import random


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                results = json.loads(line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def process_data(data, args, processed_list):
    question = data["question"]
    id = data["id"]
    if id in processed_list:
        return None
    got = GraphProcess(data, args)
    ground_truth_paths = got.get_truth_paths()
    if len(ground_truth_paths) == 0:
        return None
    if len(ground_truth_paths) > args.n_pos:
        ground_truth_paths = random.sample(ground_truth_paths, args.n_pos)

    result = {
        "id": id,
        "question": question,
        "ground_truth_paths": ground_truth_paths,
        # 'min_num_hop': min(map(len, ground_truth_paths))
    }
    return result


def main(args):
    input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.predict_path, args.d, args.save_name, args.split)
    print("Save results to: ", output_dir)

    # Load dataset
    dataset = load_dataset(
        "json", data_files=f"{os.path.join(input_file,args.split)}.jsonl"
    )["train"]

    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save args file
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    fout, processed_list = get_output_file(
        os.path.join(output_dir, "ground_truth_paths.jsonl"), force=args.force
    )
    count = 0
    if args.n == 1:
        for data in tqdm(dataset):
            res = process_data(data, args, processed_list)
            if res is None:
                count += 1
                continue
            fout.write(json.dumps(res) + "\n")
            fout.flush()
    else:
        with mp.Pool(args.n) as pool:
            for res in tqdm(
                pool.imap_unordered(
                    partial(process_data, args=args, processed_list=processed_list),
                    dataset,
                ),
                total=len(dataset),
            ):
                if res is None:
                    continue
                fout.write(json.dumps(res) + "\n")
                fout.flush()
    fout.close()
    print(f"Dont found {count}/{len(dataset)} groundtruth reasoning")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path",
        type=str,
        default="LLMReasoningCert/data",
    )
    argparser.add_argument(
        "--predict_path",
        type=str,
        default="LLMReasoningCert/data",
    )
    argparser.add_argument("--save_name", "-p", type=str, help="save name for results")
    argparser.add_argument("--seed", type=int, default=42, help="random seed")
    argparser.add_argument(
        "--model_name",
        "-m",
        type=str,
        help="save_name",
        choices=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"],
        default="gpt-3.5-turbo",
    )
    argparser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--n_pos", default=5, type=int, help="number of postive")
    argparser.add_argument(
        "-neg", default=1, type=int, help="number of negative samples"
    )
    argparser.add_argument("--debug", action="store_true")

    args = argparser.parse_args()

    if args.save_name is None:
        args.save_name = args.model_name

    utils.set_seed(args.seed)

    for d in ["cwq", "grail_qa"]:
        for split in ["test"]:
            print("Handling data {} and split {}: ".format(d, split))
            args.d = d
            args.split = split
            main(args)
