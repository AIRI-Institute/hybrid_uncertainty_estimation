import os
import yaml
import numpy as np
import json as json
import pandas as pd
from scipy.stats import rankdata
from analyze_results import rcc_auc
from ue4nlp.ue_scores import entropy as entropy_func
from ue4nlp.ue_estimator_ddu import UeEstimatorDDU
from ue4nlp.ue_estimator_rde import UeEstimatorRDE
from ue4nlp.ue_estimator_mahalanobis import UeEstimatorMahalanobis

from pathlib import Path
from tqdm.notebook import tqdm

import logging

log = logging.getLogger()


def create_ue_estimator(
    model,
    ue_args,
    eval_metric,
    calibration_dataset,
    train_dataset,
    cache_dir,
    config=None,
):
    if ue_args.ue_type == "maha":
        return UeEstimatorMahalanobis(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "ddu":
        return UeEstimatorDDU(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "rde":
        return UeEstimatorRDE(model, ue_args, config, train_dataset)
    else:
        raise ValueError()


def total_uncertainty_linear_step(
    epistemic, aleatoric, threshold_min=0.1, threshold_max=0.9, alpha=0.1
):
    n_preds = len(aleatoric)
    n_lowest = int(n_preds * threshold_min)
    n_max = int(n_preds * threshold_max)

    aleatoric_rank = rankdata(aleatoric)
    epistemic_rank = rankdata(epistemic)

    total_rank = np.zeros_like(epistemic)

    total_rank = (1 - alpha) * epistemic_rank + alpha * aleatoric_rank
    # total_rank[(aleatoric_rank > n_max)] = aleatoric_rank[(aleatoric_rank > n_max)]
    total_rank[epistemic_rank <= n_lowest] = rankdata(
        aleatoric[epistemic_rank <= n_lowest]
    )
    total_rank[
        (aleatoric_rank > n_max) & (epistemic_rank <= n_lowest)
    ] = aleatoric_rank[(aleatoric_rank > n_max) & (epistemic_rank <= n_lowest)]

    return total_rank


def read_data(path, key, ue_func):
    with open(path, "rb") as f:
        data = json.loads(f.read())

    eval_labels = np.array(data["true_labels"])
    probabilities = np.array(data["probabilities"])
    epistemic = ue_func(np.array(data[f"{key}"]))

    errors = (eval_labels != probabilities.argmax(-1)) * 1
    sr = 1 - probabilities.max(-1)
    entropy = entropy_func(probabilities)
    return eval_labels, probabilities, errors, sr, entropy, epistemic


def grid_search_hp(
    epistemic,
    aleatoric,
    errors,
    t_min_min=0.0,
    t_min_max=0.3,
    t_max_min=0.95,
    t_max_max=1.0,
    alpha_min=0.0,
    alpha_max=1.0,
):
    t_min_best = 0
    t_max_best = 1
    alpha_best = 0

    eps = 0.01
    best_rcc = rcc_auc(-epistemic, errors)
    for t_min in np.arange(t_min_min, t_min_max + eps, 0.05):
        for t_max in np.arange(t_max_min, t_max_max + eps, 0.05):
            for alpha in np.arange(alpha_min, alpha_max + eps, 0.1):
                unc = total_uncertainty_linear_step(
                    epistemic, aleatoric, t_min, t_max, alpha
                )
                new_rcc = rcc_auc(-unc, errors)
                if new_rcc < best_rcc:
                    best_rcc = new_rcc
                    t_min_best = t_min
                    t_max_best = t_max
                    alpha_best = alpha

    return best_rcc, t_min_best, t_max_best, alpha_best


def fit_method_hp(
    path,
    key,
    aleatoric,
    ue_func,
    hue_version=1,
    t_min_min=0.0,
    t_min_max=0.3,
    t_max_min=0.95,
    t_max_max=1.0,
    alpha_min=0.0,
    alpha_max=1.0,
):
    eval_labels, probabilities, errors, sr, entropy, epistemic = read_data(
        path, key, ue_func
    )

    epistemic_rcc = rcc_auc(-epistemic, errors)
    aleatoric_rcc = rcc_auc(-aleatoric, errors)

    best_rcc_before = min(epistemic_rcc, aleatoric_rcc)

    diff_before = aleatoric_rcc / epistemic_rcc

    best_rcc, t_min_best, t_max_best, alpha_best = grid_search_hp(
        epistemic,
        aleatoric,
        errors,
        t_min_min,
        t_min_max,
        t_max_min,
        t_max_max,
        alpha_min,
        alpha_max,
    )

    diff_after = best_rcc_before / best_rcc

    if hue_version == 1:
        if diff_before > diff_after * 1.6:
            t_min_best = 0
            t_max_best = 1
            alpha_best = 0

    n_preds = len(eval_labels)
    n_lowest = int(n_preds * t_min_best)
    n_max = int(n_preds * t_max_best)

    aleatoric_rank = rankdata(aleatoric)
    epistemic_rank = rankdata(epistemic)

    t1 = epistemic[epistemic_rank <= n_lowest].max() if t_min_best > 0 else 0
    t2 = aleatoric[aleatoric_rank > n_max].min() if t_max_best < 1 else 1

    return t1, t2, t_min_best, t_max_best, alpha_best, diff_before, diff_after


def fit_hybrid_hp_validation(
    dataset,
    hue_version=1,
    t_min_min=0.0,
    t_min_max=0.15,
    t_max_min=0.95,
    t_max_max=1.0,
    alpha_min=0.0,
    alpha_max=1.0,
    aleatoric_method="entropy",
    method="mahalanobis",
    key="mahalanobis_distance",
    path_val="../../workdir/run_tasks_for_model_series_method_hp/electra_raw_sn",
    ue_func=lambda x: x,
    seeds=None,
):
    if seeds is None:
        seeds = [10671619, 1084218, 23419, 42, 43, 4837, 705525]

    score_difs = []
    params = {}

    for seed in seeds:
        if aleatoric_method in ["entropy", "sr"]:
            if dataset in ["bios", "trustpilot"]:
                seed_path_val = f"{path_val}/{dataset}_miscl/0.2/{method}/results/{seed}/dev_inference.json"
            else:
                seed_path_val = f"{path_val}/{dataset}/0.2/{method}/results/{seed}/dev_inference.json"
            _, _, _, sr_val, entropy_val, _ = read_data(seed_path_val, key, ue_func)
            if aleatoric_method == "entropy":
                aleatoric = entropy_val
            else:
                aleatoric = sr_val
        else:
            seed_path_df_val = (
                f"{path_val}/{dataset}/0.2/deep_fool/results/{seed}/dev_inference.json"
            )
            seed_path_val = (
                f"{path_val}/{dataset}/0.2/{method}/results/{seed}/dev_inference.json"
            )

            _, _, _, sr_val, entropy_val, deep_fool_val = read_data(
                seed_path_df_val, "deep_fool", ue_func
            )
            aleatoric = deep_fool_val

        (
            t1,
            t2,
            t_min_best,
            t_max_best,
            alpha_best,
            diff_before,
            diff_after,
        ) = fit_method_hp(
            seed_path_val,
            key,
            aleatoric,
            ue_func,
            hue_version,
            t_min_min,
            t_min_max,
            t_max_min,
            t_max_max,
            alpha_min,
            alpha_max,
        )
        score_difs.append([diff_after, diff_before])
        params[seed] = [t1, t2, t_min_best, t_max_best, alpha_best]

    score_difs = np.array(score_difs)
    sr_better = score_difs[:, 1] < 1

    if 0 < sr_better.sum() < 2:
        # if SR better than MD, but it is outlier seed, use MD
        for seed in np.array(seeds)[sr_better != 0]:
            if hue_version == 1:
                params[seed] = [0, 1, 0, 1, 0]
            elif hue_version == 2:
                params[seed] = [-1, -1, -1, -1, 0]

    md_better = score_difs[:, 1] > 1
    if 0 < md_better.sum() < 2:
        # if MD better than SR, but it is outlier seed, use SR
        for seed in np.array(seeds)[md_better != 0]:
            if hue_version == 1:
                params[seed] = [0, 1, 0, 1, 1]
            elif hue_version == 2:
                params[seed] = [-1, -1, -1, -1, 1]

    return params
