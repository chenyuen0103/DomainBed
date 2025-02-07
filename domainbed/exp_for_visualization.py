import os
import subprocess
import time
import torch
from itertools import product


algorithms = ["ERM", "Fishr", "Fish", "CMA"]
datasets = ["ColoredMNIST", "VLCS"]
test_envs = {
    "ColoredMNIST": [0, 1, 2],
    "VLCS": [0, 1, 2, 3]
}
data_dir = "/data/common/domainbed/"

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.', flush=True)
    try:
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
        print(f'Using GPUs: {available_gpus}', flush=True)
    except Exception:
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
        print(f'Using all GPUs: {available_gpus}', flush=True)
    
    n_gpus = len(available_gpus)
    procs_by_gpu = [None] * n_gpus

    while commands:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if proc is None or proc.poll() is not None:
                # Launch a new command on this GPU
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for all processes to complete
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

def generate_commands():
    commands = []
    for dataset, algorithm in product(datasets, algorithms):
        for test_env in test_envs[dataset]:
            output_dir = f"./domainbed/uai_plot_{dataset}_{algorithm}_testenv{test_env}"
            
            # Skip if "done" file exists
            done_file = os.path.join(output_dir, "done")
            if os.path.exists(done_file):
                print(f"Skipping {output_dir} (already done)")
                continue

            cmd = (f"python3 -m domainbed.scripts.train "
                   f"--algorithm {algorithm} --dataset {dataset} "
                   f"--test_env {test_env} --data_dir={data_dir} "
                   f"--output_dir={output_dir}")
            commands.append(cmd)
    return commands

if __name__ == "__main__":
    commands = generate_commands()
    if commands:
        multi_gpu_launcher(commands)
    else:
        print("All experiments are already completed.")
