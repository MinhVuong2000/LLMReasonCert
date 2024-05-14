import os
import json
import argparse
import pandas as pd
from extract_triplet.extract_triplet import (
    load_vdb,
    search_triples_prompt,
    load_embed_model,
)
from certify_fact import certify_prompt_triples
from generative_cert.utils.utils import (
    is_correct_llm_ans_format,
    is_lack_of_knowledge,
    ans_by_sc,
    get_final_answer,
    preprocess_ans,
)


def retrieve_triplets(
    embed_model_name, device, top_k, db_path, reasoning_ans_dat, only_mean
):  # , prob_thres=prob_thres):
    print("Load embedding model")
    embed_model = load_embed_model(embed_model_name, device)
    print("Starting loading VectorDB")
    db_index, triplets = load_vdb(db_path)
    print("Corpus contains {} triplets.".format(len(triplets)))

    reasoning_ans_dat[f"{top_k}_extracted_triplets"] = reasoning_ans_dat[
        "reasoning_ans"
    ].progress_apply(
        lambda reasoning_ans: [
            search_triples_prompt(
                embed_model, db_index, triplets, ans, top_k, only_mean=only_mean
            )
            for ans in reasoning_ans
        ]
    )
    return reasoning_ans_dat


def eval(row, prob_thres, ent_thres):
    pred_ans = [get_final_answer(ans) for ans in row["reasoning_ans"]]
    ground_ans = row["groundtruth_answer"]
    extracted_triplets = row[f"{top_k}_extracted_triplets"]
    try:
        res = []
        short_res = {}
        for pred, best_triples in zip(pred_ans, extracted_triplets):
            best_tri = [dic["best_triple"].values() for dic in best_triples]
            single_res = certify_prompt_triples(
                pred, ground_ans, best_tri, prob_thres, ent_thres
            )
            res.append(single_res)
            if not short_res:
                short_res = single_res
            else:
                if (
                    single_res["p_incorrect_answer"] == 0
                    and single_res["p_incorrect_reasoning"] == 0
                ):
                    short_res = single_res
                else:
                    if (
                        single_res["p_incorrect_answer"] == 0
                        and short_res["p_incorrect_answer"] == 1
                    ):
                        short_res = single_res
        return res, short_res
    except:
        print(row)
        raise ValueError()


def save_result_df(dat, raw_dat_path, dataset, is_sc):
    version = raw_dat_path.split("/")[-2] + f"-is_sc_{is_sc}"
    model_name = raw_dat_path.split("/")[-3]
    out_dir = "LLMReasoningCert/LLMReasonCert/tmp/revision/results/"
    out_dir = os.path.join(out_dir, dataset, model_name, version)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "full.jsonl")
    dat.to_json(out_path, orient="records", lines=True)
    return out_dir


def calc_and_save_result_metrics(
    out_dir, dat, raw_dat_len, lack_knowledge_rate, incorrect_instruction_rate
):
    ans_acc = round(1 - sum(dat["p_incorrect_answer"]) / len(dat), 4) * 100
    reasoning_cert_ratio = (
        round(1 - sum(dat["p_incorrect_reasoning"]) / len(dat), 4) * 100
    )
    ## fine-grain uncert reasoning causes: fact or relation
    reasoning_uncert_by_fact_ratio = (
        round(sum(dat["p_factual_error"]) / sum(dat["p_incorrect_reasoning"]), 4) * 100
    )
    reasoning_uncert_by_relation_ratio = 100 - reasoning_uncert_by_fact_ratio

    # write
    with open(os.path.join(out_dir, "metric.txt"), "w") as fout:
        fout.write(f"lack_knowledge_rate {lack_knowledge_rate}%" + "\n")
        fout.write(f"incorrect_instruction_rate {incorrect_instruction_rate}%" + "\n")
        fout.write(f"answer_accuracy {ans_acc}%" + "\n")
        fout.write(f"reasoning_certification_ratio {reasoning_cert_ratio}%" + "\n")
        fout.write(
            f"reasoning_uncert_by_fact_ratio {reasoning_uncert_by_fact_ratio}%" + "\n"
        )
        fout.write(
            f"reasoning_uncert_by_relation_ratio {reasoning_uncert_by_relation_ratio}%"
            + "\n"
        )


def load_data(raw_dat_path, dataset, is_sc):
    # load llm generation data
    with open(raw_dat_path) as f:
        dat = [json.loads(l) for l in f]
    dat = pd.DataFrame(dat)
    dat["raw_reasoning_ans"] = dat["reasoning_ans"]
    dat["reasoning_ans"] = dat["reasoning_ans"].map(lambda x: ans_by_sc(x, is_sc))

    # if dataset is grail_qa, get ground_truth_paths
    if dataset == "grail_qa":
        splitted_ground_truth_paths_p = "LLMReasoningCert/data/grail_qa/gpt-3.5-turbo/test/splitted_ground_truth_paths.json"
        with open(splitted_ground_truth_paths_p) as f:
            origin_dat = json.load(f)
        origin_dat = origin_dat["min_2hop"] + origin_dat["min_multihop"]
        origin_dat = {v["id"]: v["ground_truth_paths"] for v in origin_dat}
        dat["ground_truth_paths"] = dat["id"].map(lambda id: origin_dat.get(id, None))
        dat = dat.dropna().reset_index()

    print(dat.head(), "\n", len(dat))
    return dat


def cert_llm_reasoning(
    raw_dat_path,
    embed_model_name,
    device,
    top_k,
    dataset,
    mode,
    is_sc,
    prob_thres,
    ent_thres,
):
    # 1. load llm generation data
    print("Loading data")
    raw_dat = load_data(raw_dat_path, dataset, is_sc)
    raw_dat_len = len(raw_dat)

    # eval1. check incorrect_instruction & lack of knowledge
    ## lack of knowledge
    print("Checking abstention")
    dat1 = raw_dat[
        raw_dat["reasoning_ans"].map(is_lack_of_knowledge) == False
    ].reset_index(drop=True)
    lack_knowledge_rate = round(1 - len(dat1) / raw_dat_len, 4) * 100
    print(dat1.head(), "\nLen:", len(dat1))
    ## incorrect_instruction
    print("Checking incorrect_instruction")
    len_dat = len(dat1)
    dat1["reasoning_ans"] = dat1["reasoning_ans"].map(
        lambda ans: [preprocess_ans(x) for x in ans]
        if isinstance(ans, list)
        else preprocess_ans(ans)
    )
    dat = dat1[dat1["reasoning_ans"].map(is_correct_llm_ans_format) == 1].reset_index(
        drop=True
    )
    incorrect_instruction_rate = round(1 - len(dat) / len_dat, 4) * 100
    print(dat.head(), "\nLen:", len(dat))

    # 2. retrieve triplets
    print("Retrieving triplets")
    db_path = f"LLMReasoningCert/data/db_extract/{dataset}/only_test_set"
    dat = retrieve_triplets(
        embed_model_name, device, top_k, db_path, dat, only_mean=True
    )

    # 3. evaluate reasoning
    # a. find GroundTruth Answer
    dat["groundtruth_answer"] = dat["ground_truth_paths"].map(
        lambda v: list({i[-1][-1] for i in v if i})
    )
    # b. eval2. eval fact, reasoning and add them to origin data
    print("Evaluating fact & reasoning")
    dat[["raw_eval", "short_eval"]] = dat.progress_apply(
        lambda r: eval(r, prob_thres, ent_thres), axis=1, result_type="expand"
    )
    dat = pd.concat([dat, pd.DataFrame(dat["short_eval"].tolist())], axis=1)

    # 4. save
    # a. save data to double check and example for writing paper
    print("Saving full data")
    out_dir = save_result_df(dat, raw_dat_path, dataset, is_sc)
    # b. calculate and save metrics
    print("Saving metrics at", out_dir)
    calc_and_save_result_metrics(
        out_dir, dat, raw_dat_len, lack_knowledge_rate, incorrect_instruction_rate
    )
    print("Done!")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset", type=str, default="cwq", choices=["cwq", "grail_qa"]
    )
    argparser.add_argument(
        "--mode",
        type=str,
        choices=["fewshot-cot-only", "fewshot-cot-hint", "fewshot-cot-hint-ground"],
    )
    argparser.add_argument("--prob_thres", type=float, default=0.7)
    argparser.add_argument("--ent_thres", type=float, default=0.5)
    argparser.add_argument("--is_sc", type=int, default=1)
    argparser.add_argument(
        "--raw_dat_path",
        type=str,
        required=True,
        help="LLMReasoningCert/data/cwq/gpt-3.5-turbo/gpt-3.5-turbo-fewshot-cot-only-cwq-temp-0.7-p-0.9-consistency-1/llm_prompt_response.jsonl",
    )
    args = argparser.parse_args()

    top_k = 10
    embed_model_name = "sentence-transformers/all-mpnet-base-v2"
    device = "cpu"
    prob_thres, ent_thres = args.prob_thres, args.ent_thres

    cert_llm_reasoning(
        args.raw_dat_path,
        embed_model_name,
        device,
        top_k,
        args.dataset,
        args.mode,
        args.is_sc,
        prob_thres,
        ent_thres,
    )
