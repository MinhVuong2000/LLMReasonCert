import json
import os
import glob
import numpy as np


result_path = "new_results_w_ans"

for data_path in glob.glob(os.path.join(result_path, "*")):
    # print(data_path)
    data_name = os.path.basename(data_path)
    eval_neg = False
    if "neg" in data_name:
        eval_neg = True
    print(f"{data_name} zero-shot zero-shot-cot few-shot few-shot-cot")
    # print(os.path.basename(data_path))
    for model_path in glob.glob(os.path.join(data_path, "*")):
        # print(model_path)
        model_name = os.path.basename(model_path)
        result_dict = {
            "zero-shot": None,
            "zero-shot-cot": None,
            "few-shot": None,
            "few-shot-cot": None,
        }
        for prediction_path in glob.glob(os.path.join(model_path, "*.jsonl")):
            total = 0
            with open(prediction_path, "r") as f:
                lines = f.readlines()
                all_result = []
                for line in lines:
                    try:
                        data = json.loads(line.strip())
                    except:
                        # print("Error in parsing line: ", line)
                        # print(prediction_path)
                        continue
                        # exit()
                    result_list = []
                    total += 1
                    for r in data["details"]:
                        response = r["raw_response"]
                        prediction = 0
                        if eval_neg:
                            if (
                                "NO" in response.upper()
                                and "YES" not in response.upper()
                            ):
                                prediction = 1
                        else:
                            if (
                                "YES" in response.upper()
                                and "NO" not in response.upper()
                            ):
                                prediction = 1
                        result_list.append(prediction)
                    all_result.append(np.mean(result_list))
                avg_result = np.mean(all_result)
                for key in result_dict.keys():
                    result_key = (
                        os.path.basename(prediction_path).split("_")[1].split(".")[0]
                    )
                    if key == result_key:
                        result_dict[key] = f"{avg_result:.4f}"
                        # result_dict[key] = f"{avg_result:.4f} ({total})"
        # for result_path in glob.glob(os.path.join(model_path, "*.txt")):
        #     # print(result_path)
        #     with open(result_path, "r") as f:
        #         lines = f.readlines()
        #         for line in lines:
        #             if "Accuracy" in line:
        #                 auc = line.split(" ")[-1].strip()
        #                 for key in result_dict.keys():
        #                     if key in result_path:
        #                         result_dict[key] = auc
        print(
            f"{model_name} {result_dict['zero-shot']} {result_dict['zero-shot-cot']} {result_dict['few-shot']} {result_dict['few-shot-cot']}"
        )
        # print("{} {} {} {} {} {} {}".format(model_name, result_dict["zero-shot"], result_dict["zero-shot-cot"], result_dict["few-shot"], result_dict["few-shot-cot"], result_dict["neg-few-shot"], result_dict["neg-few-shot-cot"]))
    print("-" * 50)
