# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

class EnvDataset(torch.utils.data.Dataset):
    """Custom dataset to include environment index with each sample."""
    def __init__(self, dataset, env_index):
        self.dataset = dataset
        self.env_index = env_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_data, label = self.dataset[idx]
        return input_data, label, self.env_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--device', type=int, default=0)
    # parser.add_argument('--data_dir', type=str, default="../data")
    parser.add_argument('--data_dir', type=str, default="/data/common/domainbed")
    parser.add_argument('--dataset', type=str, default="ColoredMNIST")
    parser.add_argument('--algorithm', type=str, default="CMA")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:",flush=True)
    print("\tPython: {}".format(sys.version.split(" ")[0]),flush = True)
    print("\tPyTorch: {}".format(torch.__version__),flush = True)
    print("\tTorchvision: {}".format(torchvision.__version__),flush = True)
    print("\tCUDA: {}".format(torch.version.cuda),flush = True)
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()),flush = True)
    print("\tNumPy: {}".format(np.__version__),flush = True)
    print("\tPIL: {}".format(PIL.__version__),flush = True)

    print('Args:',flush = True)
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v),flush = True)

    if args.hparams and 'model_type' in json.loads(args.hparams):
        model_type = json.loads(args.hparams)['model_type']
    else:
        model_type = 'ViT-S'
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset, model_type)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed), model_type)
    if args.hparams:
        hparams.update(json.loads(args.hparams))


    print('HParams:',flush = True)
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v),flush = True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = f"cuda:{args.device}"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selection method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset): # env is a dataset
        # print('Env {}: {} samples'.format(env_i, len(env)))
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        # Wrap env datasets with EnvDataset to include environment index
        in_ = EnvDataset(in_, env_i)
        out = EnvDataset(out, env_i)
        uda = EnvDataset(uda, env_i) if uda is not None and len(uda) else []


        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]
    # print(f"Training only on training environments: {[i for i in range(len(in_splits)) if i not in args.test_envs]}")

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    # breakpoint()
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None
    # breakpoint()
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        # breakpoint()
        if args.algorithm in ["CMA", "ERM", "CORAL", "Fishr", 'Fish']:
            minibatches_device = [(x.to(device), y.to(device), g.to(device)) for x,y,g in next(train_minibatches_iterator)]
        else:
            minibatches_device = [(x.to(device), y.to(device)) for x,y,_ in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        # breakpoint()
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })
            epochs_path = os.path.join(args.output_dir, 'results.jsonl')


            def inspect_data_types(data, parent_key=''):
                for key, value in data.items():
                    full_key = f"{parent_key}.{key}" if parent_key else key
                    if isinstance(value, dict):
                        inspect_data_types(value, full_key)
                    else:
                        if isinstance(value, (np.integer, np.floating, np.ndarray)):
                            print(f"Key: {full_key} | Type: {type(value)} | Value: {value}")
                        else:
                            print(f"Key: {full_key} | Type: {type(value)}")

            # Convert any NumPy float32 values to Python float before serialization
            results = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in results.items()}

            with open(epochs_path, 'a') as f:
                # inspect_data_types(results)
                # breakpoint()
                f.write(json.dumps(results, sort_keys=True) + "\n")

            # epochs_path = os.path.join(args.output_dir, args.dataset, f"{args.algorithm}_test_env_{''.join(str(env) for env in args.test_envs)}",'epochs.json')
            # if not os.path.exists(epochs_path):
            #     os.makedirs(os.path.dirname(epochs_path), exist_ok=True)
            #     with open(epochs_path, 'w') as f:
            #         f.write(json.dumps(results, sort_keys=True) + "\n")
            # with open(epochs_path, 'a') as f:
            #     f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
