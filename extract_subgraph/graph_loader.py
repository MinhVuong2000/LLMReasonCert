from generative_cert.utils.utils import *
import networkx as nx
import random

PROMPT = """Given this reasoning path, do you think this is a valid path to answer the question? If yes please answer "YES", otherwise please answer "NO".

Reasoning path:
{path}

Question:
{question}
"""


class GraphProcess(object):
    def __init__(self, data, args) -> None:
        self.args = args
        self.model = args.model_name
        self.graph = None
        self.chat_log = []
        self.memory = []
        # Init grpah and memory
        self.graph = build_graph(data["graph"])
        question = data["question"]
        if not question.endswith("?"):
            question += "?"

        self.data = data
        self.question = question

    def log(self, query, response):
        self.chat_log.append({"query": query, "response": response})

    def get_truth_paths(self):
        entities = self.data["q_entity"]
        answer_entities = self.data["a_entity"]

        # Select paths
        paths = []
        for h in entities:
            if h not in self.graph:
                continue
            for t in answer_entities:
                if t not in self.graph:
                    continue
                try:
                    for p in nx.all_shortest_paths(self.graph, h, t):
                        paths.append(p)
                except:
                    pass
        # Add relation to paths
        result_paths = []
        for p in paths:
            tmp = []
            for i in range(len(p) - 1):
                u = p[i]
                v = p[i + 1]
                tmp.append((u, self.graph[u][v]["relation"], v))
            result_paths.append(tmp)
        return result_paths

    # def get_neg_paths(self, truth_paths):
    #     '''
    #     Get negative paths
    #     '''
    #     entities = self.data['q_entity']
    #     answer_entities = self.data['a_entity']
    #     neg_paths = []
    #     for truth_path in truth_paths:
    #         l = len(truth_path)
    #         for _ in range(self.args.n_neg):

    #     for _ in range(self.args.n_neg):
    #         # Get random entity pair
    #         h = random.choice(list(self.graph.nodes))
    #         t = random.choice(list(self.graph.nodes))
    #         if h == t:
    #             continue
    #         try:
    #             for p in nx.all_shortest_paths(self.graph, h, t):
    #                 neg_paths.append(p)
    #         except:
    #             pass
    #     # Add relation to paths
    #     result_paths = []
    #     for p in neg_paths:
    #         tmp = []
    #         for i in range(len(p)-1):
    #             u = p[i]
    #             v = p[i+1]
    #             tmp.append((u, self.graph[u][v]['relation'], v))
    #         result_paths.append(tmp)
    #     return result_paths

    # def get_evaluate_data(self):
    #     truth_paths = self.get_truth_paths()
    #     if len(truth_paths) > self.args.n_pos:
    #         truth_paths = random.sample(truth_paths, self.args.n_pos)
    #     neg_paths = self.get_neg_paths(truth_paths)

    def predict_path(self, path):
        """
        Check path validity and return the answer

        Args:
            path : _description_
        """
        results = {}
        for p in path:
            path_string = path_to_string(p)
            prompt = PROMPT.format(path=path_string, question=self.question)
            response = query(prompt)
            if "YES" in response:
                results[path_string] = 1
            else:
                results[path_string] = 0
        return results
