import os
import json
# input_dir = "./results_vits_3600"
# count = 0
# if not os.path.exists("./results_vits_3600_32"):
#     os.makedirs("./results_vits_3600_32")
# for direct in os.listdir(input_dir):
#     # read the fist row of results.jsonl
#     with open(os.path.join(input_dir, direct, "results.jsonl")) as f:
#         first_line = f.readline()
#         first_line = json.loads(first_line)
#     if (first_line['hparams']['batch_size'] == 32 and 'MNIST' not in first_line['args']['dataset']):
#     # copy the direct to the new directory
#     #     os.system(f"cp -r {os.path.join(input_dir, direct)} ./results_vits_3600_32")
#         count += 1
#     elif 'MNIST' in first_line['args']['dataset'] and first_line['hparams']['batch_size'] == 64:
#         os.system(f"cp -r {os.path.join(input_dir, direct)} ./results_vits_3600_32")
#         count += 1

result_dir = './results_vits_terra_pacs'
copy_count = 0
for exp in os.listdir(result_dir):
    if not os.path.isdir(os.path.join(result_dir, exp)) or len(os.listdir(os.path.join(result_dir, exp))) == 0:
        continue
    with open(os.path.join(result_dir, exp, "results.jsonl")) as f:
        first_line = f.readline()
        first_line = json.loads(first_line)
    if (first_line['args']['algorithm'] == 'HessianAlignment') and (first_line['args']['dataset'] == 'ColoredMNIST' or first_line['args']['dataset'] == 'RotatedMNIST'):
        os.system(f"rm -r {os.path.join(result_dir, exp)}")
        # os.system(f"cp -r {os.path.join(result_dir, exp)} ./results_vits_terra_pacs")
        copy_count += 1
# print(f"Copied {copy_count} HessianAlignment experiments to ./results_vits_terra_pacs")
print(f"removed {copy_count} HessianAlignment experiments to ./results_vits_terra_pacs")



