#!/bin/bash
#SBATCH --job-name="DomainNet"
#SBATCH --output="cma.out.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bdmr-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --no-requeue
#SBATCH -t 48:00:00
#SBATCH -e slurm_logs/slurm-%j.err
#SBATCH -o slurm_logs/slurm-%j.out


python -m domainbed.scripts.sweep delete_incomplete    \
    --data_dir=/projects/bdmr/chenyuen0103/domainbed/datasets/     \
    --output_dir=/projects/bdmr/chenyuen0103/domainbed/results_resnet_domainnet     \
    --command_launcher multi_gpu       \
    --algorithms CMA       \
    --datasets DomainNet    \
    --hparams {\"model_type\":\"ResNet\"}\
    --single_test_envs       \
    --n_hparams 20       \
    --n_trials 3 \
    --skip_confirmation 

python -m domainbed.scripts.sweep launch    \
    --data_dir=/projects/bdmr/chenyuen0103/domainbed/datasets/     \
    --output_dir=/projects/bdmr/chenyuen0103/domainbed/results_resnet_domainnet    \
    --command_launcher multi_gpu       \
    --algorithms CMA       \
    --hparams {\"model_type\":\"ResNet\"}    \
    --datasets DomainNet   \
    --single_test_envs       \
    --n_hparams 20       \
    --n_trials 3 \
    --skip_confirmation 