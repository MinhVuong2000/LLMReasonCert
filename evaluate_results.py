import argparse
import json
import os
import re
import string


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"^\b(or|and)\b", "", s)
    s = re.sub(r"\"", "", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    if not prediction and not answer:
        return 1
    for a in answer:
        if match(" ".join(prediction), a):
            return 1
    for p in prediction:
        if match(" ".join(answer), p):
            return 1
    return 0


def eval_f1(prediction, answer):
    if not prediction and not answer:
        return 1, 1, 1
    if len(prediction) == 0 or len(answer) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def eval_result(predict_path, result_path, dataset, split):
    """
    Eval final answer.
    """
    predict_file = os.path.join(predict_path, "llm_prompt_response.jsonl")
    out_file = os.path.join(
        result_path, f"{dataset}_{split}_evaluate_llm_prompting.jsonl"
    )
    # Load results
    f1_list = []
    precission_list = []
    recall_list = []
    hit_list = []
    with open(predict_file) as fin, open(out_file, "w") as fout:
        # skip first line which is args
        first_line = True
        for line in fin:
            if first_line:
                first_line = False
                continue
            data = json.loads(line)
            if dataset == "grail_qa":
                groundtruth_ans = data["answer"]["entity_name"]
                if groundtruth_ans == [""]:
                    groundtruth_ans = data["answer"]["answer_argument"]
            else:
                try:
                    groundtruth_ans = list(
                        {path[-1][-1] for path in data["ground_truth_paths"] if path}
                    )
                except:
                    print(data["ground_truth_paths"])
                    raise ValueError()

            # note: havent yet handled \", or, and
            prediction_ans = re.findall(
                r"\nSo the answer is \((.*?)\)", data["reasoning_ans"]
            )
            prediction_ans = prediction_ans[0].split(", ") if prediction_ans else []
            # prediction_ans = [data['reasoning_ans']]#.split("\n")[-1].split(', ')
            f1, precision, recall = eval_f1(prediction_ans, groundtruth_ans)
            f1_list.append(f1)
            precission_list.append(precision)
            recall_list.append(recall)
            # prediction_str = ' '.join(prediction_ans)
            hit = eval_hit(prediction_ans, groundtruth_ans)
            hit_list.append(hit)
            fout.write(
                json.dumps(
                    {
                        "id": data["id"],
                        "question": data["question"],
                        "prediction": prediction_ans,
                        "ground_truth": groundtruth_ans,
                        "hit": hit,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                    }
                )
                + "\n"
            )
        result_str = {
            "Hit": str(sum(hit_list) * 100 / len(hit_list)),
            " F1": str(sum(f1_list) * 100 / len(f1_list)),
            " Precision": str(sum(precission_list) * 100 / len(precission_list)),
            " Recall": str(sum(recall_list) * 100 / len(recall_list)),
        }
        print(result_str)
        fout.write(json.dumps(result_str) + "\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-d_in",
        type=str,
        default="LLMReasoningCert/data/",
    )
    argparser.add_argument(
        "-d_out",
        type=str,
        default="LLMReasoningCert/LLMReasonCert/experiment_results/evaluate_llm_prompt",
    )
    argparser.add_argument("--dataset", type=str, default="cwq")  # grail_qa, cwq
    argparser.add_argument("--split", type=str, default="test")  # validation, test, dev
    argparser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    args = argparser.parse_args()

    args.d_in = os.path.join(args.d_in, args.dataset, args.model_name, args.split)
    eval_result(args.d_in, args.d_out, args.dataset, args.split)
