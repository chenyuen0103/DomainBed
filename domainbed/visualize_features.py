#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap.umap_ as umap
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D  # This registers the 3D projection, even if not used directly
# Import dataset and Featurizer function
from domainbed.networks import Featurizer
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms, hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
import pickle
from itertools import product

import matplotlib.patches as mpatches
import matplotlib.lines as mlines


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

def extract_features(model, dataloader, device, args={}):
    """Extract features from model for all samples in dataloader."""
    model.eval()
    features_list, labels_list, domains_list = [], [], []
    # Use zip(*dataloader) to get one batch from each loader (if using multiple loaders)
    minibatch_iterator = zip(*dataloader)
    with torch.no_grad():
        for x, y, d in next(minibatch_iterator):
            x = x.to(device)
            feats = model(x)
            features_list.append(feats.cpu().numpy())
            labels_list.append(y.numpy())
            domains_list.append(d.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    domains = np.concatenate(domains_list, axis=0)

    return features, labels, domains

def plot_features_pca(features, labels, domains, args={}):
    """Visualize features in 2D using PCA."""
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    markers = ['o', '+', 's', '^', 'D']  # Add more if you have more classes
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(8, 6))

    for d in np.unique(domains):
        for c in np.unique(labels):
            idx = (labels == c) & (domains == d)
            plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                        label=f"Env {d}, Cls {c}", alpha=0.5,  color=colors[c%len(colors)],marker=markers[d % len(markers)])
    
    plt.title(f"PCA - Feature Visualization {f'(trained) ' if args.trained else ''}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, f"{args.data}_{args.algorithm}{'_trained' if args.trained else ''}_testenv{args.test_envs[0]}_pca.pdf"), format='pdf')
    plt.show()

def plot_features_umap(features, labels, domains, args={}):
    """Visualize features in 2D using UMAP."""
    reducer = umap.UMAP(n_components=2, random_state=42)
    features_2d = reducer.fit_transform(features)
    markers = ['o', '+', 's', '^', 'D']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(8, 6))
    for d in np.unique(domains):
        for c in np.unique(labels):
            idx = (labels == c) & (domains == d)
            plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                        label=f"Env {d}, Cls {c}", alpha=0.5,  color=colors[c%len(colors)],marker=markers[d % len(markers)])
    
    plt.title(f"UMAP - Feature Visualization {f'(trained) ' if args.trained else ''}")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, f"{args.data}_{args.algorithm}{'_trained' if args.trained else ''}_testenv{args.test_envs[0]}_umap.pdf"), format='pdf')
    plt.show()

# --- New 3D Plotting Functions ---

def plot_features_pca_3d(features, labels, domains, args={}):
    """Visualize features in 3D using PCA."""
    pca = PCA(n_components=3)
    features_3d = pca.fit_transform(features)
    markers = ['o', '+', 's', '^', 'D']  # Extend this list as needed
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for d in np.unique(domains):
        for c in np.unique(labels):
            idx = (labels == c) & (domains == d)
            ax.scatter(features_3d[idx, 0], features_3d[idx, 1], features_3d[idx, 2],
                       label=f"Env {d}, Cls {c} {f'(Unseen)' if d in args.test_envs else ''}", alpha=0.5, color=colors[c%len(colors)],marker=markers[d % len(markers)])
    
    ax.set_title(f"{args.data.capitalize()}--PCA Features {f'(trained) ' if args.trained else ''}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, f"{args.data}_{args.algorithm}{'_trained' if args.trained else ''}_testenv{args.test_envs[0]}_pca3d.pdf"), format='pdf')
    plt.show()

def plot_features_umap_3d(features, labels, domains, args={}):
    """Visualize features in 3D using UMAP."""
    reducer = umap.UMAP(n_components=3, random_state=42)
    features_3d = reducer.fit_transform(features)
    markers = ['o', '+', 's', '^', 'D']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for d in np.unique(domains):
        for c in np.unique(labels):
            idx = (labels == c) & (domains == d)
            ax.scatter(features_3d[idx, 0], features_3d[idx, 1], features_3d[idx, 2],
                       label=f"Env {d}, Cls {c}", alpha=0.5, color=colors[c%len(colors)],marker=markers[d % len(markers)])
    
    ax.set_title(f"{args.data.capitalize()}--UMAP Features {f'(trained) ' if args.trained else ''}")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_zlabel("UMAP3")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, f"{args.data}_{args.algorithm}{'_trained' if args.trained else ''}_testenv{args.test_envs[0]}_umap_3d.pdf"), format='pdf')
    plt.show()

def plot_features(features, labels, domains, method='pca', dims=2, args=None):
    """
    Plots features in 2D or 3D using PCA/UMAP and 
    combines class color + domain shape in one single legend.
    """
    if args is None:
        args = {}

    # Choose dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=dims)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=dims, random_state=42)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Fit + transform
    features_reduced = reducer.fit_transform(features)

    # Decide 2D vs 3D
    if dims == 2:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    elif dims == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        raise ValueError("dims must be 2 or 3")

    # Define shapes (domain) + colors (class)
    markers = ['o', '+', 's', '^', 'D']  # shapes
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # colors

    unique_domains = np.unique(domains)
    unique_labels = np.unique(labels)

    # --- 1) Scatter plot points WITHOUT labels ---
    for d in unique_domains:
        for c in unique_labels:
            idx = (domains == d) & (labels == c)
            if dims == 2:
                ax.scatter(features_reduced[idx, 0], features_reduced[idx, 1],
                           alpha=0.5,
                           color=colors[c % len(colors)],
                           marker=markers[d % len(markers)])
            else:  # dims == 3
                ax.scatter(features_reduced[idx, 0], features_reduced[idx, 1], features_reduced[idx, 2],
                           alpha=0.5,
                           color=colors[c % len(colors)],
                           marker=markers[d % len(markers)])

    # --- 2) Build custom legend handles in one combined list ---

    # a) Class handles (colors)
    class_handles = []
    for c in unique_labels:
        color_patch = mpatches.Patch(
            color=colors[c % len(colors)],
            label=f"Class {c}"
        )
        class_handles.append(color_patch)

    # b) Domain handles (shapes)
    domain_handles = []
    for d in unique_domains:
        # A black marker is used here, but you can do something else:
        domain_marker = mlines.Line2D(
            [], [], color='black', marker=markers[d % len(markers)],
            linestyle='None', markersize=8,
            label=f"Env {d}"
        )
        domain_handles.append(domain_marker)

    # c) Combine them
    combined_handles = class_handles + domain_handles

    # d) Single legend call
    ax.legend(handles=combined_handles, title="Classes & Domains", loc='best')

    # --- 3) Title, labels, etc. ---
    method_name = method.upper()
    title_extra = "(trained) " if args.trained else ""
    if dims == 2:
        ax.set_title(f"{method_name} - 2D Feature Visualization {title_extra}")
        ax.set_xlabel(f"{method_name}1")
        ax.set_ylabel(f"{method_name}2")
    else:
        ax.set_title(f"{method_name} - 3D Feature Visualization {title_extra}")
        ax.set_xlabel(f"{method_name}1")
        ax.set_ylabel(f"{method_name}2")
        ax.set_zlabel(f"{method_name}3")

    # Build filename
    data_name = args.data
    algo_name = args.algorithm
    test_env_str = args.test_envs[0]
    suffix = '_trained' if args.trained else ''
    file_suffix = f"_{method.lower()}{dims}d"
    fig_name = f"{data_name}_{algo_name}{suffix}_testenv{test_env_str}{file_suffix}.pdf"

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, fig_name)

    plt.savefig(save_path, format='pdf')
    plt.show()

    print(f"Saved figure to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract and visualize features using PCA and UMAP")
    parser.add_argument("--data", type=str, default="VLCS",
                        choices=datasets.DATASETS,
                        help="Dataset name (e.g., ColoredMNIST, RotatedMNIST)")
    parser.add_argument("--data_dir", type=str, default="/data/common/domainbed",
                        help="Directory containing dataset")
    parser.add_argument("--save_dir", type=str, default="./plots",
                        help="Directory to save the plots")
    parser.add_argument("--algorithm", type=str, default="CMA",
                        choices=algorithms.ALGORITHMS,
                        help="Algorithm to use for feature extraction")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run model: 'cuda' or 'cpu'")
    parser.add_argument("--model_type", type=str, default="ViT-S",
                        choices=["MLP", "ViT-S"],
                        help="Type of model to use for feature extraction")
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and random_hparams).')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[3])
    parser.add_argument('--trained', action='store_true')
    parser.add_argument('--model_dir', type=str, default='')
    args = parser.parse_args()



    # if not args.model_dir:
    #     args.model_dir = f"./domainbed/uai_plot_{args.data}_{args.algorithm}_testenv{args.test_envs[0]}"
    # if not os.path.exists(os.path.join(args.model_dir, "model.pkl") and args.trained):
    #     print(f"Model not found in {args.model_dir}")
    # feature_path = f"./features/{args.data}/{args.algorithm}_testenv{args.test_envs[0]}_{'trained_' if args.trained else ''}features.pkl"
    # if os.path.exists(feature_path):
    #     save_dic = pickle.load(open(feature_path, "rb"))
    #     features = save_dic["features"]
    #     labels = save_dic["labels"]
    #     domains = save_dic["domains"]
    #     print(f"Features loaded from {feature_path}")
    # else:
    #     if not os.path.exists(f"./features/{args.data}"):
    #         os.makedirs(f"./features/{args.data}")

    #     device = args.device if torch.cuda.is_available() else "cpu"

    #     # Create the save directory if it doesn't exist
    #     os.makedirs(args.save_dir, exist_ok=True)

    #     # Load dataset using domainbed.datasets
    #     if args.data in vars(datasets):
    #         hparams = hparams_registry.default_hparams('CMA', args.data, args.model_type)
    #         dataset = vars(datasets)[args.data](args.data_dir, test_envs=[0], hparams=hparams)
    #     else:
    #         raise NotImplementedError(f"Dataset {args.data} not supported.")
    #     # Split dataset into environments for training, testing, and UDA
    #     in_splits = []
    #     out_splits = []
    #     uda_splits = []

    #     for env_i, env in enumerate(dataset):
    #         # Split dataset (in-split, out-split, and UDA-split if needed)
    #         out, in_ = misc.split_dataset(env, int(len(env) * 0.2), misc.seed_hash(args.trial_seed, env_i))
    #         in_ = EnvDataset(in_, env_i)
    #         out = EnvDataset(out, env_i)
    #         in_splits.append((in_, None))  # No weights in this example

    #     # Create InfiniteDataLoader for feature extraction
    #     train_loaders = [InfiniteDataLoader(
    #         dataset=env,
    #         weights=env_weights,
    #         batch_size=args.batch_size,
    #         num_workers=dataset.N_WORKERS)
    #         for i, (env, env_weights) in enumerate(in_splits)]
    #         # for i, (env, env_weights) in enumerate(in_splits) if i not in args.test_envs]
    #     # breakpoint()
        
    #     # Evaluate with FastDataLoader (if needed)
    #     eval_loaders = [FastDataLoader(
    #         dataset=env,
    #         batch_size=64,
    #         num_workers=dataset.N_WORKERS)
    #         for i, (env, _) in  enumerate((in_splits + out_splits + uda_splits)) if i in args.test_envs]
        
    #     input_shape = dataset.input_shape
    #     hparams = {"model_type": args.model_type}
    #     model = Featurizer(input_shape, hparams).to(device)
    #     if args.trained:
    #         model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pkl")), strict=False)

    #     # Extract features
    #     features, labels, domains = extract_features(model, train_loaders, device, args=args)

    #     save_dic = {"features": features, "labels": labels, "domains": domains}
    #     pickle.dump(save_dic, open(feature_path, "wb"))
    #     print(f"Features saved to {feature_path}")

    # plot_features(features, labels, domains, method='pca', dims=2, args=args)
    # plot_features(features, labels, domains, method='umap', dims=2, args=args)
    # plot_features(features, labels, domains, method='pca', dims=3, args=args)
    # plot_features(features, labels, domains, method='umap', dims=3, args=args)
    # 2D visualizations (existing)
    # plot_features_pca(features, labels, domains, args=args)
    # plot_features_umap(features, labels, domains, args=args)

    # # 3D visualizations (new)
    # plot_features_pca_3d(features, labels, domains, args=args)
    # plot_features_umap_3d(features, labels, domains, args=args)





    combo = product([True, False], [[0], [1], [2], [3]], ['ColoredMNIST', 'PACS','VLCS'], ['ERM','Fish','Fishr','CMA'])
    for args.trained, args.test_envs, args.data, args.algorithm in combo:
        if not args.trained and args.algorithm != 'ERM':
            continue
        feature_path = f"./features/{args.data}/{args.algorithm}_testenv{args.test_envs[0]}_{'trained_' if args.trained else ''}features.pkl"
        if not args.model_dir:
            args.model_dir = f"./domainbed/uai_plot_{args.data}_{args.algorithm}_testenv{args.test_envs[0]}"
        if not os.path.exists(os.path.join(args.model_dir, "model.pkl") and args.trained):
            print(f"Model not found in {args.model_dir}")
            continue

        
        if os.path.exists(feature_path):
            save_dic = pickle.load(open(feature_path, "rb"))
            features = save_dic["features"]
            labels = save_dic["labels"]
            domains = save_dic["domains"]
            print(f"Features loaded from {feature_path}")
        else:
            if not os.path.exists(f"./features/{args.data}"):
                os.makedirs(f"./features/{args.data}")

            device = args.device if torch.cuda.is_available() else "cpu"

            # Create the save directory if it doesn't exist
            os.makedirs(args.save_dir, exist_ok=True)

            # Load dataset using domainbed.datasets
            if args.data in vars(datasets):
                hparams = hparams_registry.default_hparams('CMA', args.data, args.model_type)
                dataset = vars(datasets)[args.data](args.data_dir, test_envs=[0], hparams=hparams)
            else:
                raise NotImplementedError(f"Dataset {args.data} not supported.")
            # Split dataset into environments for training, testing, and UDA
            in_splits = []
            out_splits = []
            uda_splits = []

            for env_i, env in enumerate(dataset):
                # Split dataset (in-split, out-split, and UDA-split if needed)
                out, in_ = misc.split_dataset(env, int(len(env) * 0.2), misc.seed_hash(args.trial_seed, env_i))
                in_ = EnvDataset(in_, env_i)
                out = EnvDataset(out, env_i)
                in_splits.append((in_, None))  # No weights in this example

            # Create InfiniteDataLoader for feature extraction
            train_loaders = [InfiniteDataLoader(
                dataset=env,
                weights=env_weights,
                batch_size=args.batch_size,
                num_workers=dataset.N_WORKERS)
                # for i, (env, env_weights) in enumerate(in_splits)]
                for i, (env, env_weights) in enumerate(in_splits) if i not in args.test_envs]
            # breakpoint()
            
            # Evaluate with FastDataLoader (if needed)
            eval_loaders = [FastDataLoader(
                dataset=env,
                batch_size=64,
                num_workers=dataset.N_WORKERS)
                for i, (env, _) in  enumerate((in_splits + out_splits + uda_splits)) if i in args.test_envs]
            
            input_shape = dataset.input_shape
            hparams = {"model_type": args.model_type}
            model = Featurizer(input_shape, hparams).to(device)
            if args.trained:
                model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pkl")), strict=False)

            # Extract features
            features, labels, domains = extract_features(model, train_loaders, device, args=args)

            save_dic = {"features": features, "labels": labels, "domains": domains}
            pickle.dump(save_dic, open(feature_path, "wb"))
            print(f"Features saved to {feature_path}")

        #                 # 2D visualizations (existing)
        plot_features(features, labels, domains, method='pca', dims=2, args=args)
        plot_features(features, labels, domains, method='umap', dims=2, args=args)
        plot_features(features, labels, domains, method='pca', dims=3, args=args)
        plot_features(features, labels, domains, method='umap', dims=3, args=args)

if __name__ == "__main__":
    main()
