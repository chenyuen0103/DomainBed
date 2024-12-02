import json
import os

import numpy as np
import tqdm
# results_dir = 'results_vits_combined_bias'
results_dir = 'results_hess_mem'


# read the results in results_dir/*/results.json

results = {}
records = []
for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(results_dir))),
                           ncols=80,
                           leave=False):
    results_path = os.path.join(results_dir, subdir, "results.jsonl")
    # breakpoint()
    try:
        with open(results_path, "r") as f:
            for line in f:
                records.append(json.loads(line[:-1]))
    except IOError:
        pass

# Group records by (dataset, algorithm)

result = {}
for r in records:
    dataset = r["args"]["dataset"]
    algorithm = r["args"]["algorithm"]
    if (dataset, algorithm) not in result:
        result[(dataset, algorithm)] = []
    result[(dataset, algorithm)].append(r)

averaged_step_time = {}
for key, records in result.items():
    averaged_step_time[key] = np.mean([r['step_time'] for r in records])


def generate_markdown_table(data):
    # Define the order of algorithms and datasets
    algorithms = ['ERM', 'CORAL', 'Fishr', 'HessianAlignment']
    datasets = ['ColoredMNIST', 'RotatedMNIST', 'VLCS', 'PACS', 'TerraIncognita']

    # Initialize the table
    table = "| Algorithm         | " + " | ".join(datasets) + " |\n"
    table += "|-------------------| " + " | ".join(['--------------' for _ in datasets]) + " |\n"

    # Fill in the table
    for algo in algorithms:
        row = f"| {algo:<17} |"
        for dataset in datasets:
            value = data.get((dataset, algo), 0)
            row += f" {value:.4f} |"
        table += row + "\n"

    return table

print(generate_markdown_table(averaged_step_time))