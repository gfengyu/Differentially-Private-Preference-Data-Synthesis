import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.decomposition import PCA
import pandas as pd
from typing import List
import torch.distributed as dist
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import json
from langdetect import detect
import argparse
import opendp.prelude as dp
dp.enable_features("contrib", "floating-point", "honest-but-curious")
import numpy as np
from opacus import PrivacyEngine
from diffprivlib.models import KMeans as DPKMeans

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('OpenAssistant')
    model_settings = parser.add_argument_group('parameter settings')
    model_settings.add_argument('--eps', type=float, default=8.0)
    model_settings.add_argument('--run', type=int, default=1,
                                help='run')
    return parser.parse_args()

def on_rank0():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

def process_oasst(dataset):
    processed_data = []
    dataset = dataset.filter(lambda x: x['lang'] == 'en')

    message_indexed_df = pd.DataFrame(dataset).set_index('message_id')
    parent_indexed_df = pd.DataFrame(dataset).set_index('parent_id')

    assistant_prefix = '\n\nAssistant: '
    assistant_suffix = ''

    human_prefix = '\n\nHuman: '
    human_suffix = ''

    def get_path_to_root(node: pd.Series):
        if node['parent_id'] is None:
            return [node]
        else:
            parent = message_indexed_df.loc[node['parent_id']]
            return [node] + get_path_to_root(parent)

    def turn_path_to_prompt(path: List[pd.Series]):
        prompt = []
        while path != []:
            node = path.pop() # earlier messages are at end of list
            prefix = assistant_prefix if node['role'] == 'assistant' else human_prefix
            suffix = assistant_suffix if node['role'] == 'assistant' else human_suffix
            prompt.append(prefix + node['text'] + suffix)

        prompt.append(assistant_prefix)
        return "".join(prompt)

    for row in (tqdm(dataset, desc='Processing OASST') if on_rank0() else dataset):
        if row['rank'] is None:
            continue

        try:
            sibling_df = parent_indexed_df.loc[row['parent_id']]
            next_best_sibling = sibling_df[sibling_df['rank'] == (row['rank'] - 1)].iloc[0]
            path_to_root = get_path_to_root(message_indexed_df.loc[next_best_sibling['message_id']])
        except KeyError:
            continue
        except IndexError:
            continue

        prompt = turn_path_to_prompt(path_to_root[1:])
        processed_data.append({
            "prompt": prompt,
            "chosen": next_best_sibling['text'] + assistant_suffix,
            "rejected": row['text'] + assistant_suffix
        })
    return processed_data

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

class RewardMLEModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

if __name__ == "__main__":
    args = parse_args()
    eps = args.eps
    run = args.run

    if eps == 8:
        eps_pca = 1
        eps_cluster = 1
        sigma_sgd = 0.471
        MAX_GRAD_NORM = 1.0
        dpsgd_batch_size = 4
        dpsgd_epochs = 4

    if eps == 4:
        eps_pca = 0.5
        eps_cluster = 0.5
        sigma_sgd = 0.566
        MAX_GRAD_NORM = 1.0
        dpsgd_batch_size = 4
        dpsgd_epochs = 4

    if eps == 2:
        eps_pca = 0.25
        eps_cluster = 0.25
        sigma_sgd = 0.671
        MAX_GRAD_NORM = 1.0
        dpsgd_batch_size = 4
        dpsgd_epochs = 4

    if eps == 1:
        eps_pca = 0.125
        eps_cluster = 0.125
        sigma_sgd = 0.808
        MAX_GRAD_NORM = 1.0
        dpsgd_batch_size = 4
        dpsgd_epochs = 4

    dataset_train = load_dataset("OpenAssistant/oasst1")['train']
    dataset_test = load_dataset("OpenAssistant/oasst1")['validation']
    processed_train = process_oasst(dataset_train)
    processed_test = process_oasst(dataset_test)

    model_name = 'BAAI/bge-large-en-v1.5'
    embedder = SentenceTransformer(model_name)

    train_chosen_texts = [f"{s['prompt']}{s['chosen']}" for s in processed_train]
    train_rejected_texts = [f"{s['prompt']}{s['rejected']}" for s in processed_train]

    train_chosen_embs = embedder.encode(train_chosen_texts, batch_size=64, show_progress_bar=True)
    train_rejected_embs = embedder.encode(train_rejected_texts, batch_size=64, show_progress_bar=True)

    embedding_differences_train = np.array(train_chosen_embs) - np.array(train_rejected_embs)

    test_chosen_texts = [f"{s['prompt']}{s['chosen']}" for s in processed_test]
    test_rejected_texts = [f"{s['prompt']}{s['rejected']}" for s in processed_test]

    test_chosen_embs = embedder.encode(test_chosen_texts, batch_size=64, show_progress_bar=True)
    test_rejected_embs = embedder.encode(test_rejected_texts, batch_size=64, show_progress_bar=True)

    embedding_differences_test = np.array(test_chosen_embs) - np.array(test_rejected_embs)

    X_train = np.stack(embedding_differences_train)
    X_test = np.stack(embedding_differences_test)

    if eps < 0:
        print("Using non-private PCA")
        pca = PCA(n_components=20)
        pca.fit(X_train)
        X_pca_train = pca.transform(X_train)
        X_pca_test = pca.transform(X_test)
    else:
        print(f"Using DP-PCA")
        dp_pca = dp.sklearn.decomposition.PCA(
            n_components=20,
            epsilon=eps_pca,
            row_norm=1.0,
            n_samples=X_train.shape[0],
            n_features=X_train.shape[1],
        )
        dp_pca.fit(X_train)
        X_pca_train = (X_train - dp_pca.mean_) @ dp_pca.components_.T
        X_pca_test = (X_test - dp_pca.mean_) @ dp_pca.components_.T

    norms = np.linalg.norm(X_pca_train, axis=1)
    X_pca_train = X_pca_train[norms > 1e-8]
    X_pca_train = X_pca_train / np.linalg.norm(X_pca_train, axis=1, keepdims=True)

    norms = np.linalg.norm(X_pca_test, axis=1)
    X_pca_test = X_pca_test[norms > 1e-8]
    X_pca_test = X_pca_test / np.linalg.norm(X_pca_test, axis=1, keepdims=True)

    num_clusters = 5
    if eps < 0:
        print("Using non-private clustering")
        kmeans = KMeans(n_clusters=num_clusters, n_init="auto")
        labels = kmeans.fit_predict(X_pca_train)
        test_labels = kmeans.predict(X_pca_test)
    else:
        print("Using DP clustering")
        dp_kmeans = DPKMeans(n_clusters=num_clusters, epsilon=eps_cluster, bounds=(-0.5, 0.5))
        dp_kmeans.fit(X_pca_train)
        labels = dp_kmeans.labels_
        test_labels = dp_kmeans.predict(X_pca_test)

    theta_list = []      # shape (num_clusters, d)
    cluster_weights = [] # length = num_clusters

    for cluster_id in range(num_clusters):
        X_cluster = X_pca_train[labels == cluster_id]
        X_test_cluster = X_pca_test[test_labels == cluster_id]

        if len(X_test_cluster) == 0:
            print(f"Cluster {cluster_id} has no test sample.")

        if len(X_cluster) < int(len(processed_train)/(num_clusters+4)):
            print(f"Cluster {cluster_id} has only {len(X_cluster)} sample(s), skip.")
            continue

        X_tensor = torch.tensor(X_cluster, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)

        model = RewardMLEModel(input_dim=X_cluster.shape[1])

        if eps < 0:
            print("Using non-private SGD")
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            num_epochs = 100
            train_losses = []
            test_losses = []
            for epoch in range(num_epochs):
                model.train()
                batch_losses = []
                for (x_batch,) in dataloader:
                    logits = model(x_batch)
                    loss = torch.nn.functional.softplus(-logits).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                train_loss = sum(batch_losses) / len(batch_losses)
                train_losses.append(train_loss)

                model.eval()
                with torch.no_grad():
                    logits_test = torch.tensor(X_test_cluster, dtype=torch.float32) @ model.linear.weight.T.squeeze()
                    test_loss = torch.nn.functional.softplus(-logits_test).mean().item()
                    test_losses.append(test_loss)
        else:
            print("Using DP-SGD")
            batch_size = dpsgd_batch_size
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            num_epochs = dpsgd_epochs

            privacy_engine = PrivacyEngine()
            model, optimizer, dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                noise_multiplier=sigma_sgd,
                max_grad_norm=MAX_GRAD_NORM,
            )

            train_losses, test_losses = [], []
            for epoch in range(num_epochs):
                model.train()
                batch_losses = []
                for (x_batch,) in dataloader:
                    logits = model(x_batch)
                    loss = torch.nn.functional.softplus(-logits).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                train_loss = sum(batch_losses) / len(batch_losses)
                train_losses.append(train_loss)

                model.eval()
                with torch.no_grad():
                    logits_test = torch.tensor(X_test_cluster, dtype=torch.float32) @ model.linear.weight.T.squeeze()
                    test_loss = torch.nn.functional.softplus(-logits_test).mean().item()
                    test_losses.append(test_loss)

        theta = model.linear.weight.detach().numpy().flatten()
        theta_list.append(theta)
        cluster_weights.append(len(X_cluster))

    public_dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
    llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
    sampling_params = SamplingParams(
        temperature=0.9,
        max_tokens=300,
        n=5,
        stop=["Human:"]
    )

    batch_size = 32
    all_synthetic_data = []

    for i in tqdm(range(0, len(public_dataset), batch_size), desc="Generating completions"):
        batch_prompts = []

        for j in range(i, min(i + batch_size, len(public_dataset))):
            prompt = f"\n\nHuman: {public_dataset[j]['instruction'].strip()} {public_dataset[j]['input'].strip()} \n\nAssistant: "
            batch_prompts.append(prompt)

        outputs = llm.generate(batch_prompts, sampling_params)
        assert len(outputs) == len(batch_prompts)

        for prompt, output in zip(batch_prompts, outputs):
            responses = [o.text.strip() for o in output.outputs]
            all_synthetic_data.append({
                "prompt": prompt,
                "responses": responses
            })

    cluster_weights = np.array(cluster_weights)
    cluster_probs = cluster_weights / cluster_weights.sum()
    theta_array = np.stack(theta_list)  # shape: (num_valid_clusters, dim)

    synthetic_preference_records = []

    # np.random.seed(42)
    assigned_clusters = np.random.choice(
        len(cluster_probs),
        size=len(all_synthetic_data),
        p=cluster_probs
    )

    for i, sample in enumerate(all_synthetic_data):
        prompt = sample['prompt']
        responses = sample["responses"]
        responses = [r for r in responses if r.strip() != ""]
        if len(responses) < 2:
            continue
        input_texts = [f"{prompt}{r}" for r in responses]
        embeddings = embedder.encode(input_texts, batch_size=16, show_progress_bar=False)

        if eps < 0:
            embeddings_pca = pca.transform(embeddings)
        else:
            embeddings_pca = (embeddings - dp_pca.mean_) @ dp_pca.components_.T

        embeddings_pca = embeddings_pca / np.linalg.norm(embeddings_pca, axis=1, keepdims=True)

        cluster_index = assigned_clusters[i]
        theta_k = theta_array[cluster_index]

        scores = embeddings_pca @ theta_k
        best_idx = int(np.argmax(scores))
        worst_idx = int(np.argmin(scores))
        if (scores[best_idx] - scores[worst_idx]) < 0.5:
            continue

        record = {
            "prompt": prompt,
            "cluster": int(cluster_index),
            "preferred": responses[best_idx],
            "unpreferred": responses[worst_idx],
            "responses": [
                {
                    "text": r,
                    "score": float(score)
                } for r, score in zip(responses, scores)
            ]
        }

        synthetic_preference_records.append(record)
    os.makedirs("./syn_OA", exist_ok=True)

    if eps < 0:
        with open(f"./syn_OA/OA_alpaca_infty_run{run}.json", "w") as f:
            json.dump(synthetic_preference_records, f, indent=2)
    else:
        with open(f"./syn_OA/OA_alpaca_eps{int(eps)}_run{run}.json", "w") as f:
            json.dump(synthetic_preference_records, f, indent=2)
