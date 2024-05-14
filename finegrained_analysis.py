import json
import pandas as pd

GPT_FOLDERS = [
    "gpt-3.5-turbo-fewshot-cot-hint-{}-temp-0.7-p-0.9-consistency-1-is_sc_1",
    "gpt-3.5-turbo-fewshot-cot-only-{}-temp-0.7-p-0.9-consistency-1-is_sc_1",
    "gpt-3.5-turbo-fewshot-cot-only-{}-temp-0.7-p-0.9-consistency-20-is_sc_4",
]
OTHER_MODELS_FOLDERS = [
    "cot-hint-temp-0.7-p-0.9-is_sc_1",
    "cot-temp-0.7-p-0.9-is_sc_1",
    "cot-temp-0.7-p-0.9-is_sc_4",
]
MODEL_LIST = [
    "gpt-3.5-turbo",
    "Llama-2-70b-chat-hf",
    "Mistral-7B-Instruct-v0.1",
    "Qwen-7B-Chat",
    "Qwen-14B-Chat",
    "vicuna-33b-v1.3",
]
DATASET_LIST = ["cwq", "grail_qa"]


def calc_avg_match_edit_rate(match_path, step_path, len_dataset):
    with open(step_path) as f:
        step_dat = [json.loads(l) for l in f]
        step_dat = [len(min(dic["10_extracted_triplets"], key=len)) for dic in step_dat]
    with open(match_path) as f:
        match_dat = [json.loads(l) for l in f]
    match_dat = pd.DataFrame(match_dat)
    len_dataset = max(len(match_dat), len_dataset)
    return {
        # 'avg_match_rate': round(match_dat['match_rate'].sum()/len_dataset,4)*100,
        # 'avg_edit_rate': round((match_dat['edit_rate'].sum()+len_dataset-len(match_dat))/len_dataset,4)*100,
        "avg_match_rate": round(match_dat["match_rate"].mean(), 4) * 100,
        "avg_edit_rate": round(match_dat["edit_rate"].mean(), 4) * 100,
        "avg_num_steps": round(sum(step_dat) / len(step_dat), 4),
    }


def check_match_edit_rate():
    path_match_template = (
        "LLMReasoningCert/LLMReasonCert/results/{}/{}/{}/groundtruth/data.jsonl"
    )
    path_step_template = "LLMReasoningCert/LLMReasonCert/results/{}/{}/{}/full.jsonl"
    res_dic = {}
    for dataset in DATASET_LIST:
        with open(
            f"LLMReasoningCert/data/{dataset}/gpt-3.5-turbo/test/splitted_ground_truth_paths.json"
        ) as f:
            dat = json.load(f)
            dat = dat["min_2hop"] + dat["min_multihop"]
            len_dataset = len(dat)
            print(len_dataset)
            # find average of steps in groundtruthpath
            dat = [s["ground_truth_paths"] for s in dat]
            num_steps = [len(min(s, key=len)) for s in dat]
            print(dataset, sum(num_steps) / len(num_steps))
        dic = {}
        for model_name in MODEL_LIST:
            model_dic = {}
            if "gpt" in model_name.lower():
                for folder in GPT_FOLDERS:
                    match_path = path_match_template.format(
                        dataset, model_name, folder.format(dataset)
                    )
                    step_path = path_step_template.format(
                        dataset, model_name, folder.format(dataset)
                    )

                    model_dic[folder] = calc_avg_match_edit_rate(
                        match_path, step_path, len_dataset
                    )
            else:
                for folder in OTHER_MODELS_FOLDERS:
                    match_path = path_match_template.format(dataset, model_name, folder)
                    step_path = path_step_template.format(dataset, model_name, folder)

                    model_dic[folder] = calc_avg_match_edit_rate(
                        match_path, step_path, len_dataset
                    )
            dic[model_name] = model_dic

        res_dic[dataset] = dic
    return res_dic


def calculate_answer_error_reasoning(path):
    with open(path) as f:
        dat = [json.loads(l) for l in f]
        dat = [dic["short_eval"] for dic in dat]
        reasoning_error_dat = [dic for dic in dat if dic["p_incorrect_reasoning"] == 1]
        fact_error_len = len(
            [dic for dic in reasoning_error_dat if dic["p_factual_error"] == 1]
        )
        coherence_error_len = len(
            [dic for dic in reasoning_error_dat if dic["p_coherent_error"] == 1]
        )
        answer_error_len = len(
            [dic for dic in reasoning_error_dat if dic["p_reasoning_ans_error"] == 1]
        )
        reasoning_error_len = len(reasoning_error_dat)
        return {
            "fact_error": round(fact_error_len / reasoning_error_len, 4) * 100,
            "coherent_error": round(coherence_error_len / reasoning_error_len, 4) * 100,
            "reasoning_answer_error": round(answer_error_len / reasoning_error_len, 4)
            * 100,
        }


def answer_error_reasoning():
    path_step_template = (
        "LLMReasoningCert/LLMReasonCert/tmp/revision/results/{}/{}/{}/full.jsonl"
    )
    res_dic = {}
    for dataset in DATASET_LIST:
        dic = {}
        for model_name in MODEL_LIST:
            model_dic = {}
            if "gpt" in model_name.lower():
                for folder in GPT_FOLDERS:
                    step_path = path_step_template.format(
                        dataset, model_name, folder.format(dataset)
                    )
                    model_dic[folder] = calculate_answer_error_reasoning(step_path)
            else:
                for folder in OTHER_MODELS_FOLDERS:
                    step_path = path_step_template.format(dataset, model_name, folder)
                    model_dic[folder] = calculate_answer_error_reasoning(step_path)
            dic[model_name] = model_dic
        res_dic[dataset] = dic
    return res_dic


if __name__ == "__main__":
    # check match, edit rate
    match_edit_dic = check_match_edit_rate()
    with open(
        "LLMReasoningCert/LLMReasonCert/tmp/revision/results/match_edit_rate.json",
        "w",
    ) as fout:
        json.dump(match_edit_dic, fout, indent=4)

    # check error
    reasoning_error_dic = answer_error_reasoning()
    with open(
        "LLMReasoningCert/LLMReasonCert/tmp/revision/results/type_error.json",
        "w",
    ) as fout:
        json.dump(reasoning_error_dic, fout, indent=4)
