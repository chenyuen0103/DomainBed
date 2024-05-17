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

result_dir = './results_vits_3600_32'
new_dir = './results_vits_VLCS_ERM_Fishr'
terr_pacs_dir = './results_vits_terra_pacs'
# if not os.path.exists(new_dir):
#     os.makedirs(new_dir)
copy_count = 0
for exp in os.listdir(terr_pacs_dir):
    if not os.path.isdir(os.path.join(terr_pacs_dir, exp)) or len(os.listdir(os.path.join(terr_pacs_dir, exp))) == 0:
        continue
    with open(os.path.join(terr_pacs_dir, exp, "results.jsonl")) as f:
        first_line = f.readline()
        first_line = json.loads(first_line)
    # if (first_line['args']['algorithm'] == 'HessianAlignment') and (first_line['args']['dataset'] == 'ColoredMNIST' or first_line['args']['dataset'] == 'RotatedMNIST'):
    if (first_line['args']['dataset'] in ['VLCS','PACS','TerraIncognita']) and (first_line['args']['algorithm'] == 'HessianAlignment'):
        os.system(f"rm -r {os.path.join(result_dir, exp)}")

        os.system(f"cp -r {os.path.join(terr_pacs_dir, exp)} {result_dir}")
        copy_count += 1
# print(f"Copied {copy_count} HessianAlignment experiments to ./results_vits_terra_pacs")
# print(f"removed {copy_count} HessianAlignment experiments to ./results_vits_terra_pacs")
print(f"moved {copy_count} VLCS PACS Terra Hessian experiments to {result_dir}")

copy_count = 0
for exp in os.listdir(new_dir):
    if not os.path.isdir(os.path.join(new_dir, exp)) or len(os.listdir(os.path.join(new_dir, exp))) == 0:
        continue
    with open(os.path.join(new_dir, exp, "results.jsonl")) as f:
        first_line = f.readline()
        try:
            first_line = json.loads(first_line)
        except:
            # print(exp)
            continue
        # first_line = json.loads(first_line)
    # if (first_line['args']['algorithm'] == 'HessianAlignment') and (first_line['args']['dataset'] == 'ColoredMNIST' or first_line['args']['dataset'] == 'RotatedMNIST'):
    if (first_line['args']['dataset'] == 'VLCS') and (first_line['args']['algorithm'] in ['ERM','Fishr']):
        os.system(f"rm -r {os.path.join(result_dir, exp)}")
        os.system(f"cp -r {os.path.join(new_dir, exp)} {result_dir}")
        copy_count += 1
# print(f"Copied {copy_count} HessianAlignment experiments to ./results_vits_terra_pacs")
# print(f"removed {copy_count} HessianAlignment experiments to ./results_vits_terra_pacs")
print(f"moved {copy_count} VLCS ERM Fishr experiments to {result_dir}")


# for exp in os.listdir(terr_pacs_dir):
#     if not os.path.isdir(os.path.join(terr_pacs_dir, exp)) or len(os.listdir(os.path.join(terr_pacs_dir, exp))) == 0:
#         continue
#     with open(os.path.join(terr_pacs_dir, exp, "results.jsonl")) as f:
#         first_line = f.readline()
#         first_line = json.loads(first_line)
#     if (first_line['args']['algorithm'] == 'HessianAlignment') and (first_line['args']['dataset'] == 'ColoredMNIST' or first_line['args']['dataset'] == 'RotatedMNIST'):
#         os.system(f"rm -r {os.path.join(terr_pacs_dir, exp)}")
#         copy_count += 1
# print(f"Removed {copy_count} ColoredMNIST and RotatedMNIST experiments from {terr_pacs_dir}")


erm_fishr_dir = './results_vits_ERM_Fishr'
# if not os.path.exists(erm_fishr_dir):
#     os.makedirs(erm_fishr_dir)
# copy_count = 0


for exp in os.listdir(result_dir):
    if not os.path.isdir(os.path.join(result_dir, exp)) or len(os.listdir(os.path.join(result_dir, exp))) == 0:
        continue
    with open(os.path.join(result_dir, exp, "results.jsonl")) as f:
        first_line = f.readline()
        first_line = json.loads(first_line)
    if (first_line['args']['algorithm'] in ['ERM','Fishr']) and (first_line['args']['dataset'] in ['RotatedMNIST','ColoredMNIST', 'PACS','TerraIncognita']):
        os.system(f"rm -r {os.path.join(result_dir, exp)}")




for exp in os.listdir(erm_fishr_dir):
    if not os.path.isdir(os.path.join(erm_fishr_dir, exp)) or len(os.listdir(os.path.join(erm_fishr_dir, exp))) == 0:
        continue
    with open(os.path.join(erm_fishr_dir, exp, "results.jsonl")) as f:
        first_line = f.readline()
        first_line = json.loads(first_line)
    if (first_line['args']['algorithm'] in ['ERM','Fishr']) and (first_line['args']['dataset'] in ['RotatedMNIST','ColoredMNIST', 'PACS','TerraIncognita']):
        os.system(f"cp -r {os.path.join(erm_fishr_dir, exp)} {result_dir}")
        copy_count += 1
print(f"Moved {copy_count} ERM Fishr experiments to {result_dir}")


combined_dir = './results_vits_combined'
new_hessian_dir = './results_vits_hessian_class'
new_hessian_MNIST_dir = './results_vits_hessian_MNIST_rescale_sqrt'
erm_fishr_dir = 'results_vits_3600_32'
terr_dir = './results_vits_hessian_class_terra'

if not os.path.exists(combined_dir):
    os.makedirs(combined_dir)

for exp in os.listdir(erm_fishr_dir):
    if not os.path.isdir(os.path.join(erm_fishr_dir, exp)) or len(os.listdir(os.path.join(erm_fishr_dir, exp))) == 0:
        continue
    with open(os.path.join(erm_fishr_dir, exp, "results.jsonl")) as f:
        first_line = f.readline()
        first_line = json.loads(first_line)
    if (first_line['args']['algorithm'] in ['ERM','Fishr']):
        os.system(f"cp -r {os.path.join(erm_fishr_dir, exp)} {combined_dir}")


for exp in os.listdir(new_hessian_dir):
    if not os.path.isdir(os.path.join(new_hessian_dir, exp)) or len(os.listdir(os.path.join(new_hessian_dir, exp))) == 0:
        continue
    with open(os.path.join(new_hessian_dir, exp, "results.jsonl")) as f:
        first_line = f.readline()
        first_line = json.loads(first_line)
    if (first_line['args']['algorithm'] == 'HessianAlignment'):
        os.system(f"cp -r {os.path.join(new_hessian_dir, exp)} {combined_dir}")

for exp in os.listdir(new_hessian_MNIST_dir):
    if not os.path.isdir(os.path.join(new_hessian_MNIST_dir, exp)) or len(os.listdir(os.path.join(new_hessian_MNIST_dir, exp))) == 0:
        continue
    with open(os.path.join(new_hessian_MNIST_dir, exp, "results.jsonl")) as f:
        first_line = f.readline()
        first_line = json.loads(first_line)
    if (first_line['args']['algorithm'] == 'HessianAlignment'):
        os.system(f"cp -r {os.path.join(new_hessian_MNIST_dir, exp)} {combined_dir}")


for exp in os.listdir(terr_dir):
    if not os.path.isdir(os.path.join(terr_dir, exp)) or len(os.listdir(os.path.join(terr_dir, exp))) == 0:
        continue
    with open(os.path.join(terr_dir, exp, "results.jsonl")) as f:
        first_line = f.readline()
        first_line = json.loads(first_line)
    if (first_line['args']['algorithm'] == 'HessianAlignment'):
        os.system(f"cp -r {os.path.join(terr_dir, exp)} {combined_dir}")











erm_fishr_mnist_dir = './results_vits_MNIST_ERM_Fishr'
if not os.path.exists(erm_fishr_mnist_dir):
    os.makedirs(erm_fishr_mnist_dir)

for exp in os.listdir(erm_fishr_dir):
    if not os.path.isdir(os.path.join(erm_fishr_dir, exp)) or len(os.listdir(os.path.join(erm_fishr_dir, exp))) == 0:
        continue
    with open(os.path.join(erm_fishr_dir, exp, "results.jsonl")) as f:
        first_line = f.readline()
        first_line = json.loads(first_line)
    if (first_line['args']['algorithm'] in ['ERM','Fishr']) and (first_line['args']['dataset'] in ['RotatedMNIST','ColoredMNIST']):
        os.system(f"cp -r {os.path.join(erm_fishr_dir, exp)} {erm_fishr_mnist_dir}")
        copy_count += 1
print(f"Moved {copy_count} MNIST ERM Fishr experiments to {erm_fishr_mnist_dir}")



erm_fishr_pacs_dir = './results_vits_PACS_Terra_ERM_Fishr'
if not os.path.exists(erm_fishr_pacs_dir):
    os.makedirs(erm_fishr_pacs_dir)

for exp in os.listdir(erm_fishr_dir):
    if not os.path.isdir(os.path.join(erm_fishr_dir, exp)) or len(os.listdir(os.path.join(erm_fishr_dir, exp))) == 0:
        continue
    with open(os.path.join(erm_fishr_dir, exp, "results.jsonl")) as f:
        first_line = f.readline()
        first_line = json.loads(first_line)
    if (first_line['args']['algorithm'] in ['ERM','Fishr']) and (first_line['args']['dataset'] in ['RotatedMNIST','ColoredMNIST']):
        os.system(f"cp -r {os.path.join(erm_fishr_dir, exp)} {erm_fishr_pacs_dir}")
        copy_count += 1
print(f"Moved {copy_count} MNIST ERM Fishr experiments to {erm_fishr_pacs_dir}")



