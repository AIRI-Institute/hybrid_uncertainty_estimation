import torch
import copy
import gc
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

from utils.utils_heads import (
    ElectraClassificationHeadIdentityPooler,
    BertClassificationHeadIdentityPooler,
    ElectraNERHeadIdentityPooler,
)
from utils.utils_inference import is_custom_head, unpad_features, pad_scores

import logging
import time

log = logging.getLogger()


DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10**exp for exp in range(-10, 0, 1)]


def centered_cov(x):
    return x.T @ x / (len(x) - 1)


def compute_density(log_logits, label_probs):
    return torch.logsumexp(log_logits, dim=1)
    # return torch.logsumexp(log_logits*label_probs[label_probs>0], dim=1)
    # return torch.sum((torch.exp(log_logits / 768) * label_probs[label_probs>0]), dim=1)


def get_gmm_log_probs(gaussians_model, embeddings):
    return gaussians_model.log_prob(embeddings[:, None, :])


def gmm_fit(embeddings, labels):
    num_classes = len(set(labels))
    with torch.no_grad():
        centroids = torch.stack(
            [torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)]
        )
        cov_matrix = torch.stack(
            [
                centered_cov(embeddings[labels == c] - centroids[c])
                for c in range(num_classes)
            ]
        )

    with torch.no_grad():
        gmm = None
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    cov_matrix.shape[1],
                    device=cov_matrix.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=centroids,
                    covariance_matrix=(cov_matrix + jitter),
                )
                break
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "The parameter covariance_matrix has invalid values" in str(e):
                    continue
        if gmm is None:
            cov_matrix_eye = torch.eye(
                cov_matrix.shape[1],
                device=cov_matrix.device,
            ).unsqueeze(0)
            gmm = torch.distributions.MultivariateNormal(
                loc=centroids,
                covariance_matrix=cov_matrix_eye,
            )

    return gmm, jitter_eps


class UeEstimatorDDU:
    def __init__(self, cls, ue_args, config, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset

    def __call__(self, X, y):
        return self._predict_with_fitted_gmm(X, y)

    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model

        self.use_tanh = "use_tanh" in self.ue_args.keys() and self.ue_args.use_tanh
        self.use_encoder_feats = (
            "use_encoder_feats" in self.ue_args.keys()
            and self.ue_args.use_encoder_feats
        )
        self.return_train_preds = (
            "return_train_preds" in self.ue_args.keys()
            and self.ue_args.return_train_preds
        )
        log.info("****************Start fitting GMM**************")

        if y is None:
            y = self._exctract_labels(X)

        if self.return_train_preds:
            self.train_preds = self._exctract_preds(X)
            self.train_labels = y

        self._replace_model_head()
        X_features = self._exctract_features(X)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_features = torch.Tensor(X_features).to(self.device)

        self.gmm, jitter = gmm_fit(X_features, y)
        self.label_probs = torch.Tensor(np.bincount(y) / len(y)).to(self.device)

        if self.return_train_preds:
            log_probs = get_gmm_log_probs(self.gmm, X_features)
            self.train_scores = (
                compute_density(log_probs, self.label_probs)
                .cpu()
                .detach()
                .numpy()
                .tolist()
            )

        # assert torch.all(self.label_probs > 0), "All labels must present in the train sample!"

        log.info("**************Done.**********************")

    def _replace_model_head(self):
        log.info("Change classifier to Identity Pooler")
        cls = self.cls
        model = self.cls._auto_model

        if is_custom_head(model):
            model.classifier = ElectraClassificationHeadIdentityPooler(model.classifier)
            if self.use_encoder_feats:
                model.classifier.dense = torch.nn.Identity()
                model.classifier.activation = torch.nn.Identity()
                model.classifier.dropout1 = torch.nn.Identity()
            if self.use_tanh:
                model.classifier.activation = torch.nn.Tanh()
        else:
            if "fairlib" in self.config.model.model_name_or_path:
                if "mlp" in self.config.model.model_name_or_path:
                    model.output_layer = torch.nn.Identity()
                else:
                    model.classifier.output_layer = torch.nn.Identity()

                if "inlp" in self.config.model.model_name_or_path:
                    model.return_hiddens = True
            else:
                model.classifier = BertClassificationHeadIdentityPooler(
                    model.classifier
                )
                if "distilbert" in self.config.model.model_name_or_path:
                    if self.use_encoder_feats:
                        model.pre_classifier = torch.nn.Identity()
                        model.pre_classifier_activation = torch.nn.Identity()
                        model.dropout = torch.nn.Identity()
                    if self.use_tanh:
                        model.pre_classifier_activation = torch.nn.Tanh()
                elif "deberta" in self.config.model.model_name_or_path:
                    if self.use_tanh:
                        model.pooler.activation = torch.nn.Tanh()

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
        X_features = np.array(X_features, np.float64)

        return X_features

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

    def _predict_with_fitted_gmm(self, X, y):
        cls = self.cls
        model = self.cls._auto_model

        log.info("****************Compute DDU with fitted GMM**************")

        start = time.time()
        if y is None:
            y = self._exctract_labels(X)

        X_features = self._exctract_features(X)
        X_features = torch.Tensor(X_features).to(self.device)

        if X_features.shape[0] > 20000:
            # use batched ddu
            BATCH_SIZE = 512
            scores_batch = []
            for i in tqdm(range(X_features.shape[0] // BATCH_SIZE + 1)):
                torch.cuda.empty_cache()
                X_features_batch = X_features[BATCH_SIZE * i : BATCH_SIZE * (i + 1)]

                log_probs = get_gmm_log_probs(self.gmm, X_features_batch)
                score = compute_density(log_probs, None).cpu().detach().numpy()
                scores_batch.append(score)
                del X_features_batch
                gc.collect()

            scores = np.concatenate(scores_batch)
        else:
            log_probs = get_gmm_log_probs(self.gmm, X_features)
            scores = compute_density(log_probs, self.label_probs)
            scores = scores.cpu().detach().numpy()
        end = time.time()

        eval_results = {}
        eval_results["ddu_scores"] = scores.tolist()

        if self.return_train_preds:
            eval_results["train_ddu_scores"] = self.train_scores
            eval_results["train_preds"] = self.train_preds.tolist()
            eval_results["train_labels"] = self.train_labels.flatten().tolist()

        sum_inf_time = end - start
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")

        log.info("**************Done.**********************")
        return eval_results


class UeEstimatorDDUNer:
    def __init__(self, cls, ue_args, config, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset

    def __call__(self, X, y):
        return self._predict_with_fitted_gmm(X, y)

    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model

        log.info("****************Start fitting GMM**************")

        if y is None:
            y, y_shape = self._exctract_labels(X)

        self._replace_model_head()
        X_features = self._exctract_features(X)
        X_features, y = unpad_features(X_features, y)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_features = torch.Tensor(X_features).to(self.device)

        self.gmm, jitter = gmm_fit(X_features, y)

        self.label_probs = torch.Tensor(np.bincount(y) / len(y)).to(self.device)
        assert torch.all(
            self.label_probs > 0
        ), "All labels must present in the train sample!"

        log.info("**************Done.**********************")

    def _replace_model_head(self):
        log.info("Change classifier to Identity Pooler")
        cls = self.cls
        model = self.cls._auto_model

        if is_custom_head(model):
            model.classifier = ElectraNERHeadIdentityPooler(model.classifier)
        else:
            model.classifier = BertClassificationHeadIdentityPooler(model.classifier)

    def _exctract_labels(self, X):
        y = np.asarray([example["labels"] for example in X])
        y_shape = y.shape

        return y.reshape(-1), y_shape

    def _exctract_features(self, X):
        cls = self.cls
        model = self.cls._auto_model

        try:
            X = X.remove_columns("labels")
        except:
            X.dataset = X.dataset.remove_columns("labels")

        X_features = cls.predict(X, apply_softmax=False, return_preds=False)[0]
        X_features = X_features.reshape(-1, X_features.shape[-1])

        return X_features

    def _predict_with_fitted_gmm(self, X, y):
        cls = self.cls
        model = self.cls._auto_model

        log.info("****************Compute DDU with fitted GMM**************")

        start = time.time()

        y_pad, y_shape = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        X_features, y = unpad_features(X_features, y_pad)
        X_features = torch.Tensor(X_features).to(self.device)

        log_probs = get_gmm_log_probs(self.gmm, X_features)
        scores = compute_density(log_probs, self.label_probs)
        scores = pad_scores(
            scores.cpu().detach().numpy(), np.asarray(y_pad).reshape(y_shape), y_pad
        )

        end = time.time()

        eval_results = {}
        eval_results["ddu_scores"] = scores.tolist()

        sum_inf_time = end - start
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")

        log.info("**************Done.**********************")
        return eval_results
