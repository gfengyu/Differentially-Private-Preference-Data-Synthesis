from prv_accountant import PRVAccountant
from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism, GaussianMechanism
import numpy as np


dataset = "OA"
assert dataset in ["summarize", "HH", "OA"]
eps_pca_list = []
eps_cluster_list = []
sigma_sgd_list = []
sigma_histogram_list = []

if dataset == "OA":
    full_train_num = 14167
    n_cluster = 5
    cluster_size = int(full_train_num/(n_cluster+4))
    n = 4
    running_steps_dpsgd = int(4/(n/cluster_size)) # 4 epochs
    eps_pca_list = [0.125, 0.25, 0.5, 1]
    eps_cluster_list = [0.125, 0.25, 0.5, 1]
    sigma_sgd_list = [0.808, 0.671, 0.566, 0.471]

elif dataset == "HH":
    full_train_num = 160800
    n_cluster = 5
    cluster_size = int(full_train_num/(n_cluster+4))
    n = 4
    running_steps_dpsgd = int(4/(n/cluster_size)) # 4 epochs
    eps_pca_list = [0.125, 0.25, 0.5, 1]
    eps_cluster_list = [0.125, 0.25, 0.5, 1]
    sigma_sgd_list = [0.620, 0.556, 0.487, 0.412]

elif dataset == "summarize":
    full_train_num = 92858
    n_cluster = 5
    cluster_size = int(full_train_num/(n_cluster+4))
    n = 4
    running_steps_dpsgd = int(4/(n/cluster_size)) # 4 epochs
    eps_pca_list = [0.125, 0.25, 0.5, 1]
    eps_cluster_list = [0.125, 0.25, 0.5, 1]
    sigma_sgd_list = [0.647, 0.575, 0.501, 0.422]

sample_rate = n / cluster_size
assert len(eps_pca_list) == len(sigma_sgd_list) == len(eps_cluster_list)
delta = 1 / full_train_num

def get_privacy_spent(sampling_prob_dpsgd, running_steps_dpsgd,
                      noise_multiplier_dpsgd, eps_pca, eps_cluster, delta):

    prv_dpsgd = PoissonSubsampledGaussianMechanism(
        noise_multiplier=noise_multiplier_dpsgd,
        sampling_probability=sampling_prob_dpsgd,
    )
    accountant = PRVAccountant(
        prvs=[prv_dpsgd],
        max_self_compositions=[running_steps_dpsgd],
        eps_error=0.01,
        delta_error=delta/10,
    )

    eps_lower, eps_estimate, eps_upper = accountant.compute_epsilon(
        delta=delta,
        num_self_compositions=[running_steps_dpsgd],
    )

    return eps_upper + eps_pca + eps_cluster

for i in range(len(eps_pca_list)):
    eps_pca = eps_pca_list[i]
    eps_cluster = eps_cluster_list[i]
    sigma_sgd = sigma_sgd_list[i]

    total_eps = get_privacy_spent(
        sampling_prob_dpsgd=sample_rate,
        running_steps_dpsgd=running_steps_dpsgd,
        noise_multiplier_dpsgd=sigma_sgd,
        eps_pca=eps_pca,
        eps_cluster=eps_cluster,
        delta=delta
    )

    print(f"total ε = {total_eps:.3f}")



