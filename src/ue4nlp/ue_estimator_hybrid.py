import torch
import numpy as np
from tqdm import tqdm
import time
import os
import json
from sklearn.metrics import roc_auc_score

from ue4nlp.utils_hybrid_ue import (
    total_uncertainty_linear_step,
    fit_hybrid_hp_validation,
    create_ue_estimator,
)

from utils.utils_heads import (
    ElectraClassificationHeadIdentityPooler,
    BertClassificationHeadIdentityPooler,
    ElectraNERHeadIdentityPooler,
)
from utils.utils_inference import is_custom_head, unpad_features, pad_scores

import copy
from scipy.stats import rankdata
import logging

log = logging.getLogger()


class BertClassificationHeadIdentityPooler(torch.nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, other=None):
        super().__init__()

    def forward(self, pooled_output, labels=None):
        return pooled_output  # , torch.rand((pooled_output.shape[0], pooled_output.shape[0]))


def entropy(x):
    return np.sum(-x * np.log(np.clip(x, 1e-8, 1)), axis=-1)


def deepfool(x, net, max_iter=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.to(device)
    x = torch.from_numpy(x).to(device)

    x_pert = torch.clone(x).detach().to(device)
    x_pert.requires_grad_()

    preds_orig = net(x)[0]
    num_classes = preds_orig.shape[0]
    label = preds_orig.data.cpu().numpy().flatten().argmax()

    input_shape = x.detach().cpu().numpy().shape
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    preds_pert = net(x_pert)[0]
    k_i = label

    while k_i == label and loop_i < max_iter:
        pert = np.inf
        preds_pert[label].backward(retain_graph=True)
        grad_orig = x_pert.grad.data.cpu().numpy().copy()

        for k in range(num_classes):
            if k == label:
                continue

            x_pert.grad.data.zero_()

            preds_pert[k].backward(retain_graph=True)
            cur_grad = x_pert.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (preds_pert[k] - preds_pert[label]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        x_pert = x.to(device) + torch.from_numpy(r_tot).to(device)
        x_pert = torch.clone(x_pert).detach().to(device)
        x_pert.requires_grad_()

        preds_pert = net(x_pert)[0]
        k_i = np.argmax(preds_pert.data.cpu().numpy().flatten())

        loop_i += 1

    return (r_tot * r_tot).sum()


class UeEstimatorHybrid:
    def __init__(self, cls, ue_args, config, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset

    def __call__(self, X, y):
        return self._predict_with_fitted_estimator(X, y)

    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        self.model = copy.deepcopy(self.cls._auto_model)

        log.info("****************Start fitting epistmic UE**************")

        self.epistemic_ue_estimator = create_ue_estimator(
            self.cls,
            self.config.ue.epistemic,
            eval_metric=None,
            calibration_dataset=None,
            train_dataset=self.train_dataset,
            cache_dir=self.config.cache_dir,
            config=self.config,
        )

        self.X_train = X

        self.epistemic_ue_estimator.fit_ue(X)
        self.params = {}
        for hue_version in [1]:
            self.params[hue_version] = self._fit_hp(hue_version=hue_version)

        log.info("**************Done.**********************")

    def _exctract_labels(self, X):
        return np.asarray([example["label"] for example in X])

    def _exctract_preds(self, X):
        cls = self.cls
        model = self.cls._auto_model
        X_cp = copy.deepcopy(X)

        try:
            X_cp = X_cp.remove_columns("label")
        except:
            X_cp.dataset = X_cp.dataset.remove_columns("label")

        X_preds = cls.predict(X_cp, apply_softmax=True, return_preds=False)[0]
        return X_preds

    def _fit_hp(self, X=None, hue_version=1):
        log.info("****************Start fitting HP for Hybrid UE**************")
        base_path = "/".join(self.config.output_dir.split("/")[:-2])
        hp_path = (
            f"{base_path}/hybrid_{hue_version}_hp_{self.ue_args.epistemic.ue_type}.json"
        )

        if self.ue_args.epistemic.ue_type == "ddu":
            self.ue_func = lambda x: -x
        else:
            self.ue_func = lambda x: x

        if self.ue_args.epistemic.ue_type == "maha":
            method = "mahalanobis"
        else:
            method = self.ue_args.epistemic.ue_type
        if os.path.exists(hp_path):
            with open(hp_path, "r") as f:
                params = json.loads(f.read())
        else:
            if self.config.data.task_name in ["trustpilot", "bios"]:
                seeds = os.listdir(
                    f"{self.ue_args.val_path}/{self.config.data.task_name}_miscl/0.2/{method}/results"
                )
            else:
                seeds = os.listdir(
                    f"{self.ue_args.val_path}/{self.config.data.task_name}/0.2/{method}/results"
                )
            params = fit_hybrid_hp_validation(
                self.config.data.task_name,
                hue_version=hue_version,
                aleatoric_method=self.ue_args.aleatoric,
                t_min_min=self.ue_args[f"v{hue_version}"].t_min_min,
                t_min_max=self.ue_args[f"v{hue_version}"].t_min_max,
                t_max_min=self.ue_args[f"v{hue_version}"].t_max_min,
                t_max_max=self.ue_args[f"v{hue_version}"].t_max_max,
                alpha_min=self.ue_args[f"v{hue_version}"].alpha_min,
                alpha_max=self.ue_args[f"v{hue_version}"].alpha_max,
                key=self.ue_args.epistemic_key,
                method=method,
                path_val=self.ue_args.val_path,
                ue_func=self.ue_func,
                seeds=seeds,
            )
            with open(hp_path, "w") as f:
                json.dump(params, f)
        return params

    def _replace_model_head(self):
        log.info("Change classifier to Identity Pooler")
        cls = self.cls
        model = self.cls._auto_model
        self.old_head = copy.deepcopy(model.classifier)

        model.classifier = BertClassificationHeadIdentityPooler(model.classifier)

    def _return_head(self):
        self.cls._auto_model.classifier = self.old_head
        log.info("Change Identity Pooler to classifier")

    def _exctract_labels(self, X):
        return np.asarray([example["label"] for example in X])

    def _exctract_features(self, X):
        cls = self.cls
        model = self.cls._auto_model

        try:
            X = X.remove_columns("label")
        except:
            X.dataset = X.dataset.remove_columns("label")

        X_features = cls.predict(X, apply_softmax=False, return_preds=False)[0]
        return X_features

    def _predict_with_fitted_estimator(self, X, y, hue_versions=[1]):
        cls = self.cls
        model = self.model

        log.info("****************Compute Hybrid uncertainty **************")

        epistemic_res = self.epistemic_ue_estimator(X, y)
        epistemic = np.array(epistemic_res[self.ue_args.epistemic_key])

        try:
            epistemic_time = epistemic_res["ue_time"]
        except:
            epistemic_time = 0

        epistemic = self.ue_func(epistemic)

        self.cls._trainer.model = self.model
        self.cls._auto_model = self.model

        if self.ue_args.aleatoric == "entropy":
            probs = self._exctract_preds(X)
            start = time.time()
            aleatoric = entropy(probs)
        elif self.ue_args.aleatoric == "deep_fool":
            log.info("****************Compute DeepFool dists**************")

            self._replace_model_head()
            head_copy = copy.deepcopy(self.old_head)
            X_encoder_features = self._exctract_features(X)

            start = time.time()
            aleatoric = np.zeros(X_encoder_features.shape[0])
            for i, x in tqdm(enumerate(X_encoder_features)):
                if len(x.shape) == 2:
                    aleatoric[i] = -deepfool(x[None, :, :], head_copy)
                else:
                    aleatoric[i] = -deepfool(x[None, :], head_copy)

            log.info("****************Done.**************")
        elif self.ue_args.aleatoric == "sr":
            probs = self._exctract_preds(X)
            start = time.time()
            aleatoric = 1 - probs.max(-1)

        eval_results = {}
        eval_results["epistemic"] = epistemic.tolist()
        eval_results["aleatoric"] = aleatoric.tolist()

        for hue_version in hue_versions:
            t1, t2, t_min_best, t_max_best, alpha_best = self.params[hue_version][
                str(self.config.seed)
            ]

            t1_best = (epistemic <= t1).mean() if t_min_best > 0 else 0.0
            t2_best = (aleatoric < t2).mean() if t_max_best < 1 else 1.0

            t1_best = np.clip(t1_best, self.ue_args[f"v{hue_version}"].t_min_min, 0.3)
            t2_best = np.clip(
                t2_best,
                self.ue_args[f"v{hue_version}"].t_max_min,
                self.ue_args[f"v{hue_version}"].t_max_max,
            )

            hue = total_uncertainty_linear_step(
                epistemic, aleatoric, t1_best, t2_best, alpha_best
            )
            end_1 = time.time()

            eval_results[f"hue_uncertainty_{hue_version}"] = hue.tolist()

        sum_inf_time = epistemic_time + (end_1 - start)
        eval_results["ue_time"] = sum_inf_time
        eval_results["ue_epistemic_time"] = epistemic_time
        log.info("**************Done.**********************")
        return eval_results
