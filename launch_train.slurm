#!/bin/bash
#SBATCH --job-name="OfficeHome_train"
#SBATCH --output="cma.out.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16 
#SBATCH --gpus=1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1    # Request only 1 GPU
#SBATCH --gpu-bind=closest   # Select a CPU close to the GPU on PCI bus topology
#SBATCH --account=bdmr-delta-gpu    # Match to a "Project" returned by the "accounts" command
#SBATCH --no-requeue
#SBATCH -t 8:00:00
#SBATCH -e slurm_logs/slurm-%j.err
#SBATCH -o slurm_logs/slurm-%j.out



python3 -m domainbed.scripts.train   --algorithm CMA   --hparams '{"model_type":"ResNet"}'   --dataset DomainNet --test_env 2   --data_dir=/projects/bdmr/chenyuen0103/domainbed/datasets   --output_dir=/projects/bdmr/chenyuen0103/domainbed/resnet_test 

