import sys
import os
import json
from tasks.superglue.finetune import default_metrics

root_path = sys.argv[1]
best_value, best_result, best_name = None, None, None
for entry in os.scandir(root_path):
    if entry.is_dir():
        result_file = os.path.join(entry.path, "results.json")
        config_file = os.path.join(entry.path, "config.json")
        with open(config_file) as file:
                config = json.load(file)
        task = config["task"].lower()
        metric = default_metrics[task][0][0]
        valid_result, test_result = None, None
        if os.path.exists(result_file):
            with open(result_file) as file:
                lines = file.readlines()
            if len(lines) > 1:
                valid_result = json.loads(lines[0].strip())
                test_result = json.loads(lines[1].strip())
        if test_result is not None:
            metric_value = valid_result[metric]
            if best_value is None or metric_value > best_value:
                best_value = metric_value
                best_result = valid_result
                best_name = entry.name
        else:
            print(f"{entry.name} not completed yet")
print(f"Best result found at {best_name}: {best_result}")