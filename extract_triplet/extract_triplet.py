import faiss
import numpy as np
from faiss import write_index, read_index, normalize_L2
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import argparse
import torch
import os
import pickle
import itertools
import time
import json
from tqdm import tqdm
from fuzzywuzzy import fuzz
from generative_cert.utils.utils import (
    drop_duplicated_triplets,
    is_unknown_ent,
    get_unknown_ent_cates,
    is_correct_llm_ans_format,
)


def load_embed_model(pretrained_model_name, device):
    # Instantiate the sentence-level
    model = SentenceTransformer(pretrained_model_name)
    # Check if GPU is available and use it
    if torch.cuda.is_available() and ("cuda" in device):
        model = model.to(device)
    return model


def skip_unknow_ent(triplets, verbose=True):
    if verbose:
        print("Creating dict of unknow entities:0-list & 2-list...")
    unknow_ent_cates = get_unknown_ent_cates(triplets)

    if verbose:
        print("len(unknow_ent_cates)", len(unknow_ent_cates))
        print(unknow_ent_cates["m.0ydts6l"])
        print("Skipping triplets containing unknow entities...")
    new_triplets = []
    normal_trip_count = 0
    for tri in tqdm(triplets):
        head, rel, tail = tri
        if is_unknown_ent(head):
            l = unknow_ent_cates[head].get(2, [])
            new_list = [
                [t[0], rel, tail]
                for t in l
                if (
                    tail != t[0]
                    and rel.split(".")[:-1] == t[1].split(".")[:-1]
                    and rel.split(".")[-1] != t[1].split(".")[-1]
                )
            ]
            new_triplets += new_list
        elif is_unknown_ent(tail):
            l = unknow_ent_cates[tail].get(0, [])
            new_list = [
                [head, rel, t[2]]
                for t in l
                if (
                    head != t[2]
                    and rel.split(".")[:-1] == t[1].split(".")[:-1]
                    and rel.split(".")[-1] != t[1].split(".")[-1]
                )
            ]
            new_triplets += new_list
        else:
            new_triplets += [tri]
            normal_trip_count += 1
    print("normal_trip_count", normal_trip_count)
    print("num new_triplets", len(new_triplets))
    # drop duplicates
    print("drop duplicates")
    new_triplets = drop_duplicated_triplets(new_triplets)
    # new_new_triplets = []
    # print('drop duplicates')
    # for tri in tqdm(new_triplets):
    #     if (
    #         (tri[::-1] not in new_new_triplets)
    #           and (tri not in new_new_triplets)
    #     ):
    #         new_new_triplets += [tri]
    return new_triplets


def only_rel_from_triplets(triplets, verbose=True):
    if verbose:
        print("Creating dict of relations..")
    dic = {}
    for tri in tqdm(triplets):
        rel = tri[1]
        dic[rel] = [tri] if rel not in dic else dic[rel] + [tri]
    dic = {rel: drop_duplicated_triplets(dic[rel]) for rel in dic}
    return dic


def create_data_from_dataset(data_dir, our_dir, KG="cwq", verbose=True):
    origin_path = os.path.join(our_dir, "origin", "triplets.pkl")
    skip_path = os.path.join(our_dir, "skip_unknown_ent", "triplets.pkl")
    rel_path = os.path.join(our_dir, "only_relation", "triplets.pkl")
    if os.path.exists(origin_path):
        if verbose:
            print("Found origin_path triplets corpus. Reloading...")
        with open(skip_path, "rb") as f:
            triplets = pickle.load(f)
        return triplets

    # get all subgraph from dataset
    triplets = []
    data = KG
    splits = ["test.jsonl"]
    for split in splits:
        if verbose:
            print("Handling ", os.path.join(data_dir, data, split))
        dat = load_dataset("json", data_files=os.path.join(data_dir, data, split))[
            "train"
        ]
        triplets = triplets + list(itertools.chain(*dat["graph"]))
        # drop duplicates
        triplets = drop_duplicated_triplets(triplets)
    # normalize: remove useless space
    triplets = [[tri[0].strip(), tri[1], tri[2].strip()] for tri in triplets]
    # save sentences
    with open(origin_path, "wb") as f:
        pickle.dump(triplets, f)
    if verbose:
        print("Saved triplets corpus at: ", origin_path)
        print(
            f"Origin Corpus skipped unknown entities contains {len(triplets)} triplets."
        )

    ###########SKIP_UNKNOWN_ENTITIES#############
    # with open(os.path.join(args.vector_db_path,'triplets.pkl'), "rb") as f:
    #     triplets = pickle.load(f)
    # new_triplets = skip_unknow_ent(triplets, verbose=verbose)
    # if verbose:
    #     print(f'New Corpus skipped unknown entities contains {len(new_triplets)} triplets.')
    #     print(new_triplets[1])
    # with open(skip_path, "wb") as f:
    #     pickle.dump(new_triplets, f)
    ##########

    ###########ONLY RELATION#############
    # with open(os.path.join(args.vector_db_path,'skip_unknown_ent','triplets.pkl'), "rb") as f:
    #     new_triplets = pickle.load(f)
    # if verbose:
    #     print(f'Corpus skipped unknown entities contains {len(new_triplets)} triplets.')
    #     print(new_triplets[1])
    # rel_dic = only_rel_from_triplets(new_triplets, verbose=verbose)
    # with open(rel_path, "wb") as f:
    #     pickle.dump(rel_dic, f)
    # if args.verbose:
    #     print(f'Corpus contains {len(rel_dic)} relations.')
    ##########

    return triplets


def create_vdb(triplets, embed_model, path_dir, verbose):
    # embedding triplet_sents
    # 1. triplets to sentences
    triplet_sents = [
        f"{triplet[0]} {triplet[1]} {triplet[2]}."
        for triplet in triplets
        if len(triplet) == 3
    ]
    # triplet_sents = [f'The relation between {triplet[0]} and {triplet[2]} is {triplet[1]}.' for triplet in triplets]
    # 2. embed
    embeddings = embed_model.encode(
        triplet_sents, batch_size=256, show_progress_bar=True, convert_to_tensor=True
    )
    # save embedding
    torch.save(embeddings, os.path.join(path_dir, "embedding.pt"))
    if verbose:
        print("Saved embeddings at: ", os.path.join(path_dir, "embedding.pt"))

    # index using faiss
    # 1.Change data type
    embeddings = np.array([embedding for embedding in embeddings]).astype("float32")
    normalize_L2(embeddings)
    # 2.Instantiate the index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    # 3.Pass the index to IndexIDMap
    index = faiss.IndexIDMap(index)
    # 4.Add vectors and their IDs
    index.add_with_ids(embeddings, range(len(embeddings)))
    # save vector db to file
    write_index(index, os.path.join(path_dir, "faiss.index"))
    if verbose:
        print("Saved faiss database at: ", os.path.join(path_dir, "faiss.index"))

    return index


def load_vdb(path_dir):
    # load vector db from file
    db_index = read_index(os.path.join(path_dir, "faiss.index"))
    # load sentences
    with open(os.path.join(path_dir, "origin", "triplets.pkl"), "rb") as f:
        triplets = pickle.load(f)
    return db_index, triplets


def search(embed_model, db_index, triplets, sentences, top_k):
    if isinstance(sentences, str):
        sentences = [sentences]
    query_embed = embed_model.encode(sentences, batch_size=64, show_progress_bar=False)
    query_embed = np.array(query_embed).astype("float32")
    normalize_L2(query_embed)
    D, I = db_index.search(query_embed, k=top_k)
    res = []
    for sent, probs, idxs in zip(sentences, D, I):
        probs = probs.round(4).tolist()
        tris = [triplets[i] for i in idxs]
        probs_tris = list(zip(probs, tris))
        res.append((sent, probs_tris))
    return res


def scoring(sent, triplet, prob):
    # def overlap_ent(sent, ent):
    #     words = ent.split(' ')
    #     overlap_words = [w for w in words if w in sent]
    #     return len(overlap_words)/words
    ent1_score = fuzz.partial_ratio(sent, triplet[0]) / 100
    ent2_score = fuzz.partial_ratio(sent, triplet[2]) / 100
    return {
        "triplet": triplet,
        "prob": prob,
        "head_score": ent1_score,
        "tail_score": ent2_score,
    }


def search_triples_prompt(
    embed_model,
    db_index,
    triplets,
    reasoning_ans,
    top_k,
    only_mean=False,
    prob_thres=None,
):
    sent_res = []
    if is_correct_llm_ans_format(reasoning_ans):
        reasoning_ans = reasoning_ans.split("\n")[:-1]  # skip so the answer ...

        #  solve newline not a reasoning step
        new_reasoning_ans = []
        last_sent = ""
        for line in reasoning_ans:
            if not last_sent:
                last_sent = line
            else:
                if line[0].isdigit():
                    new_reasoning_ans.append(last_sent)
                    last_sent = line
                else:
                    last_sent = "\n".join([last_sent, line])
        new_reasoning_ans.append(last_sent)

        sent = [sent[3:] for sent in new_reasoning_ans]  # remove '<number>. '
        search_res = search(embed_model, db_index, triplets, sent, top_k)
        sent_res = [
            {
                "sentence": sent,
                "triplets": [scoring(sent, tri, prob) for (prob, tri) in prob_tri],
            }
            for (sent, prob_tri) in search_res
        ]

        if only_mean:
            for i, dic in enumerate(sent_res):
                mean_scores = [
                    (dic["prob"] + dic["head_score"] + dic["tail_score"]) / 3
                    for dic in dic["triplets"]
                ]
                if prob_thres:
                    top_triples = [
                        dic["triplets"][i]
                        for i in range(len(mean_scores))
                        if mean_scores[i] >= prob_thres
                    ]
                    sent_res[i] = {
                        "sentence": dic["sentence"],
                        "top_triples": top_triples,
                    }
                else:
                    max_idx = mean_scores.index(max(mean_scores))
                    sent_res[i] = {
                        "sentence": dic["sentence"],
                        "best_triple": dic["triplets"][max_idx],
                    }
    return sent_res


def search_data_split(embed_model, db_index, triplets, data_path, top_k):
    with open(data_path) as f:
        dic_list = [json.loads(line) for line in f]
    dic_list = dic_list[1:]  # ignore first line: args
    res_list = []
    for dic in tqdm(
        dic_list,
        desc=data_path.replace("LLMReasoningCert/data/", ""),
        total=len(dic_list),
    ):
        reasoning_ans = dic["reasoning_ans"]
        sent_res = search_triples_prompt(
            embed_model, db_index, triplets, reasoning_ans, top_k
        )
        res = {"id": dic["id"], f"{top_k}_extracted_triplets": sent_res}
        res_list.append(res)
    return res_list


def main(args):
    if args.verbose:
        print("Loading embedding model")
    embed_model = load_embed_model(args.embed_model_name, args.device)

    st = time.time()
    if args.create_db:
        if args.verbose:
            print("Starting creating VectorDB")
        triplets = create_data_from_dataset(
            args.triplets_path, args.vector_db_path, args.dataset, args.verbose
        )
        if args.verbose:
            print("Corpus contains {} triplets.".format(len(triplets)))

        triplets = [[tri[0].strip(), tri[1], tri[2].strip()] for tri in triplets]
        db_index = create_vdb(triplets, embed_model, args.vector_db_path, args.verbose)
    else:
        if args.verbose:
            print("Starting loading VectorDB")
        db_index, triplets = load_vdb(args.vector_db_path)
        triplets = [[tri[0].strip(), tri[1], tri[2].strip()] for tri in triplets]
        if args.verbose:
            print("Corpus contains {} triplets.".format(len(triplets)))
    if args.verbose:
        print(f"Took {time.time()-st}s to load Database!")
        # print(f'Starting searching top-{args.top_k} similar triplet...')

    # st = time.time()
    # if args.query:
    #     _, prob, res = search(embed_model, db_index, triplets, args.query, args.top_k)[0]
    #     print("Result: \n\t", args.query, '=', res, '\n\twith probability=',prob)
    #     if args.verbose:
    #         print(f'Done searching. Took {time.time()-st}s!')
    # else:
    #     datasets = [KG]
    #     splits = ['test']
    #     if args.verbose:
    #         print(f'Seaching all question in {splits} in datasets {datasets}')
    #     for data in datasets:
    #         for split in splits:
    #             data_path = f'LLMReasoningCert/data/{data}/gpt-3.5-turbo/{split}/llm_prompt_response.jsonl'
    #             out_path = data_path.replace('llm_prompt_response.jsonl','only_test_extract_triplet_skip_unknown_ent.jsonl')
    #             print('Out: ', out_path)
    #             res_list = search_data_split(embed_model, db_index, triplets, data_path, args.top_k)
    #             with open(out_path, 'w') as fout:
    #                  fout.write(json.dumps({'args':args.__dict__}) + '\n')
    #                  for res in res_list:
    #                      fout.write(json.dumps(res) + '\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--query", type=str, help="a sentence from LLM", default="")
    argparser.add_argument(
        "--origin_question",
        type=str,
        help="question to ask LLM which create the query",
        default="",
    )
    argparser.add_argument(
        "--embed_model_name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
    )
    argparser.add_argument("--device", type=str, default="cpu")
    argparser.add_argument("--top_k", type=int, default=10)
    argparser.add_argument(
        "--triplets_path",
        type=str,
        help="path to create triplets",
        default="LLMReasoningCert/data",
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        help="dataset to create KG",
        default="grail_qa",
        choices=["cwq", "grail_qa"],
    )
    argparser.add_argument(
        "--vector_db_path",
        type=str,
        help="path to load vectordb",
        default="LLMReasoningCert/data/db_extract/{}/only_test_set",
    )
    argparser.add_argument(
        "--create_db", help="create embedding of triplets or not", action="store_true"
    )
    argparser.add_argument("--verbose", help="print or not", action="store_false")

    args = argparser.parse_args()

    args.vector_db_path = args.vector_db_path.format(args.dataset)

    main(args)
