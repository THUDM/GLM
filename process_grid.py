import sys
import os
import json
import glob
import statistics

path_pattern = sys.argv[1]
target_type = sys.argv[2]
best_value, best_result, best_name = None, None, None
mean_result = {}
print(path_pattern)
for entry in glob.glob(path_pattern, recursive=True):
    valid_result, test_found = None, False
    with open(entry) as file:
        for line in file:
            data = json.loads(line.strip())
            if data["type"] == target_type:
                valid_result = data
            if "epoch" not in data:
                test_found = True
    if not test_found:
        print(f"{entry} not completed yet")
    if sys.argv[3] == "max":
        metric = sys.argv[4]
        metric_value = valid_result[metric]
        if best_value is None or metric_value > best_value:
            best_value = metric_value
            best_result = valid_result
            best_name = entry
    elif sys.argv[3] == "mean" or sys.argv[3] == "median":
        if mean_result:
            for metric, value in valid_result.items():
                if metric not in ["type", "epoch"]:
                    mean_result[metric].append(value)
        else:
            mean_result = {metric: [value] for metric, value in valid_result.items() if
                           metric not in ["type", "epoch"]}

if sys.argv[3] == "max":
    print(f"Best result found at {best_name}: {best_result}")
    with open(best_name) as file:
        print(file.read())
elif sys.argv[3] == "mean":
    mean_result = {metric: sum(value) / len(value) for metric, value in mean_result.items()}
    print(f"Mean result {mean_result}")
elif sys.argv[3] == "median":
    mean_result = {metric: statistics.median(value) for metric, value in mean_result.items()}
    print(f"Mean result {mean_result}")
