# import openai
# from dotenv import load_dotenv
import time
import os
import networkx as nx

# import tiktoken
import random
from collections import deque
import numpy as np
import re
import itertools
import statistics as stat

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.organization = os.getenv("OPENAI_ORG")
# os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G


def get_edge_data(G: nx.Graph, start_node, end_node):
    relation = G.get_edge_data(start_node, end_node)["relation"]
    return relation


# function for obtaining answer using self-consistency or not
def ans_by_sc(ans, is_sc):
    """
    ans: str or list of str
    is_sc: int. Self-consistency if is_sc>1, otherwise NO
    """
    if not isinstance(is_sc, int):
        raise ValueError("is_sc only can be INT")
    if isinstance(ans, str):
        if is_sc > 1:
            raise ValueError("Only 1 answer!")
        return [ans]
    if is_sc > 1:
        ans = ans[:is_sc]
        ans = filter_majority_vote_ans(ans)
        return ans
    else:
        return ans[:1]


def preprocess_ans(str_ans):
    str_ans = str_ans.strip()
    str_ans = str_ans.replace("\n\n", "\n")
    str_ans = str_ans.replace("Step ", "").replace("step ", "")
    str_ans = str_ans[str_ans.find("1. ") :]  # remove Q: A:
    # str_ans = str_ans[:str_ans.find('Q: ')] # remove Q: A:
    return str_ans.strip()


def is_correct_llm_ans_format(ans):
    # if not all(element in ans for element in ["So the answer is (", "1. "]):
    #     return False
    check_single = lambda ans: (
        re.search("^1. ", ans)
        and re.findall("(t|T)he answer( to the question)? is\:? \(?(.*?)\)?.?$", ans)
    )
    if isinstance(ans, str):
        return check_single(ans)
    return any(check_single(a) for a in ans)


def is_lack_of_knowledge(ans):
    keywords = [
        "do not have knowledge",
        "not have knowledge",
        "more information",
        "need more",
        "impossible",
        "not possible",
        "unknown",
        "no answer",
        "unable",
        "cannot",
        "sorry",
        "unclear",
        # 'depend on',
        # 'need to'
    ]
    check_single = lambda a: any(kw in a.lower() for kw in keywords)
    if isinstance(ans, str):
        return check_single(ans)
    return all(check_single(a) for a in ans)


def get_final_answer(str_ans):
    if isinstance(str_ans, str):
        prediction_ans = re.findall(
            r"(t|T)he answer( to the question)? is\:? \(?(.*?)\)?.?$", str_ans
        )
        if not prediction_ans:
            return []
        else:
            pred_ans = prediction_ans[0][2].split(", ")
            return pred_ans
    else:
        raise ValueError("Get final answer: need to be a string!")


def filter_majority_vote_ans(ans):
    """
    Only select answer containing majority voted answer.
    """
    if isinstance(ans, str):
        raise ValueError("Answer need to be a list of strings!")
    final_ans = [get_final_answer(a) for a in ans]  # get final answer
    list_ans = list(set(itertools.chain(*final_ans)))
    if not list_ans:
        return ans  # incorrect instruction
    major_ans = stat.mode(list_ans)
    filtered_ans = [a for a, fa in zip(ans, final_ans) if major_ans in fa]
    return filtered_ans


def drop_duplicated_triplets(tri_list):
    return list(k for k, _ in itertools.groupby(sorted(tri_list)) if len(k) == 3)


def is_unknown_ent(ent):
    if re.search("^[mg]\.", ent):
        return True
    return False


def find_triplets_contain_unknown_ent(triplets, ent, position):
    # find list of triplets which unknown ent is at the position
    l = list(filter(lambda t: t[position] == ent, triplets))
    return l


def get_unknown_ent_cates(path):
    """Creating dict of unknow entities:0-list & 2-list..."""
    unknown_ent_cates = {}
    for tri in path:
        head, rel, tail = tri
        if is_unknown_ent(head):
            if head not in unknown_ent_cates:
                unknown_ent_cates[head] = {0: [tri]}
            else:
                unknown_ent_cates[head][0] = (
                    unknown_ent_cates[head][0] + [tri]
                    if unknown_ent_cates[head].get(0, None)
                    else [tri]
                )
        elif is_unknown_ent(tail):
            if tail not in unknown_ent_cates:
                unknown_ent_cates[tail] = {2: [tri]}
            else:
                unknown_ent_cates[tail][2] = (
                    unknown_ent_cates[tail][2] + [tri]
                    if unknown_ent_cates[tail].get(2, None)
                    else [tri]
                )
    return unknown_ent_cates


def processed_groundtruth_path(path):
    """skip unknown entities."""
    # find dic of unknown entities
    unknown_ent_cates = get_unknown_ent_cates(path)
    if not unknown_ent_cates:
        return path
    new_triplets = []
    # merge
    for tri in path:
        head, rel, tail = tri
        if is_unknown_ent(head):
            l = unknown_ent_cates[head].get(2, [])
            new_list = [[t[0], "/".join([t[1], rel]), tail] for t in l if tail != t[0]]
            new_triplets += new_list
        elif is_unknown_ent(tail):
            l = unknown_ent_cates[tail].get(0, [])
            new_list = [[head, "/".join([rel, t[1]]), t[2]] for t in l if head != t[2]]
            new_triplets += new_list
        else:
            new_triplets += [tri]
    # drop duplicates
    new_triplets = drop_duplicated_triplets(new_triplets)
    return new_triplets
