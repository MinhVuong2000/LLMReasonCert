import openai
from dotenv import load_dotenv
import time
import os
import networkx as nx
import tiktoken
import random
from collections import deque
import numpy as np

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")
os.environ["TIKTOKEN_CACHE_DIR"] = "./tmp"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G


# Define a function to perform a breadth-first search
def bfs_with_rule(graph, start_node, target_rule):
    result_paths = []
    queue = deque(
        [(start_node, [])]
    )  # Use queues to store nodes to be explored and corresponding paths
    while queue:
        current_node, current_path = queue.popleft()

        # If the current path matches the rules, add it to the results list
        if len(current_path) == len(target_rule):
            result_paths.append(current_path)

        # If the current path length is less than the rule length, continue exploring
        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                # Pruning: If the current edge type does not match the corresponding position in the rule, the path will not be explored further.
                rel = graph[current_node][neighbor]["relation"]
                if rel != target_rule[len(current_path)] or len(current_path) > len(
                    target_rule
                ):
                    continue
                queue.append((neighbor, current_path + [(current_node, rel, neighbor)]))

    return result_paths


def random_walk_edge_sampling(graph, start_node, path_length, num_paths):
    paths = []

    for _ in range(num_paths):
        path = [start_node]
        current_node = start_node
        edges = []  # Used to record the edges of each path

        for _ in range(path_length - 1):
            neighbors = list(graph.neighbors(current_node))

            if len(neighbors) == 0:
                break

            next_node = random.choice(neighbors)
            edges.append(graph[current_node][next_node]["relation"])
            path.append(next_node)
            current_node = next_node

        paths.append(edges)

    return paths


def list_to_string(l: list) -> str:
    prompt = '"{}"'
    return ", ".join([prompt.format(i) for i in l])


def rule_to_string(rule: list) -> str:
    return " -> ".join(rule)


def rules_to_string(rules: list) -> str:
    prompt = []
    for r in rules:
        rule_to_string(r)
    return "\n".join(prompt)


def path_to_string(path: list) -> str:
    result = ""
    for i, p in enumerate(path):
        h, r, t = p
        result += f"Step {i+1}: {h} -> {r} -> {t}\n"

    return result.strip()


def reoder_path_to_string(path: list) -> str:
    result = ""
    for i, p in enumerate(path):
        h, r, t = p
        result += f"Step {i+1}: {h} -> {r} -> {t}\n"

    return result.strip()


def num_tokens_from_message(path_string, model):
    """Returns the number of tokens used by a list of messages."""
    messages = [{"role": "user", "content": path_string}]
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"]:
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
    elif model == "gpt-4":
        tokens_per_message = 3
    else:
        raise NotImplementedError(
            f"num_tokens_from_messages() is not implemented for model {model}."
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_token_limit(model="gpt-4"):
    """Returns the token limitation of provided model"""
    if model in ["gpt-4", "gpt-4-0613"]:
        num_tokens_limit = 8192
    elif model in ["gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"]:
        num_tokens_limit = 16384
    elif model in [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613",
        "text-davinci-003",
        "text-davinci-002",
    ]:
        num_tokens_limit = 4096
    else:
        raise NotImplementedError(
            f"""get_token_limit() is not implemented for model {model}."""
        )
    return num_tokens_limit


def split_path_list(path_list, token_limit, model):
    """
    Split the path list into several lists, each list can be fed into the model.
    """
    output_list = []
    current_list = []
    current_token_count = 4

    for path in path_list:
        path += "\n"
        path_token_count = num_tokens_from_message(path, model) - 4
        if (
            current_token_count + path_token_count > token_limit
        ):  # If the path makes the current list exceed the token limit
            output_list.append(current_list)
            current_list = [path]  # Start a new list.
            current_token_count = path_token_count + 4
        else:  # The new path fits into the current list without exceeding the limit
            current_list.append(path)  # Just add it there.
            current_token_count += path_token_count
    # Add the last list of tokens, if it's non-empty.
    if current_list:  # The last list not exceed the limit but no more paths
        output_list.append(current_list)
    return output_list


def query(message, model="gpt-3.5-turbo"):
    """
    Query ChatGPT API
    :param message:
    :return:
    """
    prompt_len = num_tokens_from_message(message, model)
    token_limitation = get_token_limit(model)
    if prompt_len > token_limitation:
        raise ValueError(
            f"Message length {prompt_len} exceeds token limitation {token_limitation}. Message: {message}"
        )
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                request_timeout=60,
                temperature=0.0,
                stream=False,
            )
            return response["choices"][0]["message"]["content"].strip()
            # collected_chunks = []
            # collected_messages = []
            # # iterate through the stream of events
            # for chunk in response:
            #     collected_chunks.append(chunk)  # save the event response
            #     chunk_message = chunk['choices'][0]['delta']  # extract the message
            #     collected_messages.append(chunk_message)  # save the message
            # result = ''.join([m.get('content', '') for m in collected_messages]).strip()
            # return result
        except Exception as e:
            print(e)
            time.sleep(60)
            continue
