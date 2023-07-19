import torch
import copy
import numpy as np
from tqdm import tqdm
import time
from sklearn.decomposition import KernelPCA
from sklearn.covariance import MinCovDet
from sklearn.model_selection import StratifiedKFold

from utils.utils_heads import (
    ElectraClassificationHeadIdentityPooler,
    BertClassificationHeadIdentityPooler,
    ElectraNERHeadIdentityPooler,
)
from utils.utils_inference import is_custom_head, unpad_features, pad_scores
from ue4nlp.mahalanobis_distance import (
    mahalanobis_distance,
    mahalanobis_distance_relative,
    mahalanobis_distance_marginal,
    compute_centroids,
    compute_covariance,
)

import logging
import copy

log = logging.getLogger()


def MCD_covariance(X, y, label, seed):
    try:
        if label == None:
            cov = MinCovDet(random_state=seed).fit(X)
        else:
            cov = MinCovDet(random_state=seed).fit(X[y == label])
    except:
        log.info(
            "****************Try fitting covariance with support_fraction=0.9 **************"
        )
        try:
            if label == None:
                cov = MinCovDet(random_state=seed, support_fraction=0.9).fit(X)
            else:
                cov = MinCovDet(random_state=seed, support_fraction=0.9).fit(
                    X[y == label]
                )
        except:
            log.info(
                "****************Try fitting covariance with support_fraction=1.0 **************"
            )
            if label == None:
                cov = MinCovDet(random_state=seed, support_fraction=1.0).fit(X)
            else:
                cov = MinCovDet(random_state=seed, support_fraction=1.0).fit(
                    X[y == label]
                )
    return cov


class UeEstimatorRDE:
    def __init__(self, cls, ue_args, config, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset

    def __call__(self, X, y):
        return self._predict_with_fitted_cov(X, y)

    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model

        self.use_pca = "use_pca" in self.ue_args.keys() and self.ue_args.use_pca
        self.use_mcd = "use_mcd" in self.ue_args.keys() and self.ue_args.use_mcd
        self.use_class_cov = (
            "use_class_cov" in self.ue_args.keys() and self.ue_args.use_class_cov
        )
        self.select_predict_dists = (
            "select_predict_dists" in self.ue_args.keys()
            and self.ue_args.select_predict_dists
        )
        self.return_train_preds = (
            "return_train_preds" in self.ue_args.keys()
            and self.ue_args.return_train_preds
        )
        self.use_tanh = "use_tanh" in self.ue_args.keys() and self.ue_args.use_tanh
        self.use_encoder_feats = (
            "use_encoder_feats" in self.ue_args.keys()
            and self.ue_args.use_encoder_feats
        )

        log.info(
            "****************Start fitting covariance and centroids **************"
        )

        if y is None:
            y = self._exctract_labels(X)

        if self.return_train_preds:
            self.train_preds = self._exctract_preds(X)
            self.train_labels = y

        self._replace_model_head()
        X_features = self._exctract_features(X)
        self.x_features = X_features
        if self.use_pca:
            print("fit pca")
            self.pca = KernelPCA(
                n_components=100, kernel="rbf", random_state=self.config.seed
            )
            n_samples = X_features.shape[0]
            if n_samples > 5000:
                n_folds = max(n_samples // 5000, 2)
                skf = StratifiedKFold(n_splits=n_folds)
                train_index, test_index = list(skf.split(X_features, y))[0]
                self.pca.fit(X_features[test_index])
                X_pca = self.pca.transform(X_features)
            else:
                X_pca = self.pca.fit_transform(X_features)
        else:
            X_pca = X_features

        if self.use_mcd:
            print("fit mcd")
            if self.use_class_cov:
                self.covariances = [
                    MCD_covariance(X_pca, y, label, self.config.seed)
                    for label in np.unique(y)
                ]
            else:
                self.covariances = [
                    MCD_covariance(X_pca, y, label=None, seed=self.config.seed)
                ]
        else:
            self.centroids = compute_centroids(X_pca, y, True)
            if self.use_class_cov:
                self.covariances = [
                    compute_covariance(
                        self.centroids[None, label, :],
                        X_pca[y == label],
                        y[y == label],
                        True,
                    )
                    for label in np.unique(y)
                ]
            else:
                self.covariances = [compute_covariance(self.centroids, X_pca, y, True)]

        if self.return_train_preds:
            self.train_features = X_pca

        log.info("**************Done.**********************")

    def _return_head(self):
        if "fairlib" in self.config.model.model_name_or_path:
            if "mlp" in self.config.model.model_name_or_path:
                self.cls._auto_model.output_layer = self.old_head
            else:
                self.cls._auto_model.classifier.output_layer = self.old_head
        else:
            self.cls._auto_model.classifier = self.old_head
        log.info("Change Identity Pooler to classifier")

    def _replace_model_head(self):
        log.info("Change classifier to Identity Pooler")
        cls = self.cls
        model = self.cls._auto_model
        if "fairlib" in self.config.model.model_name_or_path:
            if "mlp" in self.config.model.model_name_or_path:
                self.old_head = copy.deepcopy(model.output_layer)
            else:
                self.old_head = copy.deepcopy(model.classifier.output_layer)
        else:
            self.old_head = copy.deepcopy(model.classifier)

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

    def _exctract_features(self, X, return_preds=False):
        cls = self.cls
        model = self.cls._auto_model

        try:
            if "label" in X.column_names:
                X = X.remove_columns("label")
        except:
            if "label" in X.dataset.column_names:
                X.dataset = X.dataset.remove_columns("label")

        X_features = cls.predict(X, apply_softmax=False, return_preds=return_preds)[0]
        X_features = np.array(X_features, np.float64)

        return X_features

    def _predict_with_fitted_cov(self, X, y):
        cls = self.cls
        model = self.cls._auto_model

        log.info(
            "****************Compute MD with fitted covariance and centroids **************"
        )

        start = time.time()
        if y is None:
            y = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        end = time.time()

        eval_results = {}

        if self.use_pca:
            X_pca = self.pca.transform(X_features)
        else:
            X_pca = X_features

        self._return_head()
        preds = self._exctract_features(X, return_preds=True)
        preds = np.array(preds, np.int64)

        self._replace_model_head()

        if self.use_mcd:
            all_dists = np.array([cov.mahalanobis(X_pca) for cov in self.covariances]).T
            if self.return_train_preds:
                train_dists = np.array(
                    [cov.mahalanobis(self.train_features) for cov in self.covariances]
                ).T
        else:
            all_dists = np.array(
                [
                    mahalanobis_distance(
                        None, None, X_pca, self.centroids[None, i, :], cov
                    )[0]
                    for i, cov in enumerate(self.covariances)
                ]
            ).T

        if self.select_predict_dists:
            if len(self.covariances) == 1:
                md = all_dists.flatten()
            else:
                md = np.array(
                    [dist_i[pred_i] for pred_i, dist_i in zip(preds, all_dists)]
                )

                if self.return_train_preds:
                    train_preds = self.train_preds.argmax(-1)
                    train_md = np.array(
                        [
                            dist_i[pred_i]
                            for pred_i, dist_i in zip(train_preds, train_dists)
                        ]
                    )
        else:
            md = all_dists.min(-1)

        sum_inf_time = end - start
        eval_results["mahalanobis_distance"] = md.tolist()
        eval_results["ue_time"] = sum_inf_time

        if self.return_train_preds:
            eval_results["train_mahalanobis_distance"] = train_md.tolist()
            eval_results["train_preds"] = self.train_preds.tolist()
            eval_results["train_labels"] = self.train_labels.flatten().tolist()

        log.info(f"UE time: {sum_inf_time}")

        log.info("**************Done.**********************")
        return eval_results


class UeEstimatorRDENer:
    def __init__(self, cls, ue_args, config, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset

    def __call__(self, X, y):
        return self._predict_with_fitted_cov(X, y)

    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model

        log.info(
            "****************Start fitting covariance and centroids **************"
        )

        if y is None:
            y, y_shape = self._exctract_labels(X)

        self._replace_model_head()
        X_features = self._exctract_features(X)
        X_features, y = unpad_features(X_features, y)

        self.pca = KernelPCA(
            n_components=100, kernel="rbf", random_state=self.config.seed
        )
        X_pca = self.pca.fit_transform(X_features)
        self.covariances = [
            MinCovDet(random_state=self.config.seed).fit(X_pca[y == label])
            for label in np.unique(y)
        ]

        log.info("**************Done.**********************")

    def _fit_covariance(self, X, y, class_cond=True):
        if class_cond:
            return compute_covariance(self.class_cond_centroids, X, y, class_cond)
        return compute_covariance(self.train_centroid, X, y, class_cond)

    def _fit_centroids(self, X, y, class_cond=True):
        return compute_centroids(X, y, class_cond)

    def _return_head(self):
        self.cls._auto_model.classifier = self.old_head
        log.info("Change Identity Pooler to classifier")

    def _replace_model_head(self):
        log.info("Change classifier to Identity Pooler")
        cls = self.cls
        model = self.cls._auto_model
        self.old_head = copy.deepcopy(model.classifier)

        if is_custom_head(model):
            model.classifier = ElectraNERHeadIdentityPooler(model.classifier)
        else:
            model.classifier = BertClassificationHeadIdentityPooler(model.classifier)

    def _exctract_labels(self, X):
        y = np.asarray([example["labels"] for example in X])
        y_shape = y.shape

        return y.reshape(-1), y_shape

    def _exctract_features(self, X, return_preds=False):
        cls = self.cls
        model = self.cls._auto_model

        try:
            X = X.remove_columns("labels")
        except:
            X.dataset = X.dataset.remove_columns("labels")

        X_features = cls.predict(X, apply_softmax=False, return_preds=return_preds)[0]
        X_features = X_features.reshape(-1, X_features.shape[-1])

        return X_features

    def _predict_with_fitted_cov(self, X, y):
        cls = self.cls
        model = self.cls._auto_model

        log.info(
            "****************Compute MD with fitted covariance and centroids **************"
        )

        start = time.time()

        y_pad, y_shape = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        X_features, y = unpad_features(X_features, y_pad)

        end = time.time()

        eval_results = {}

        self._return_head()
        preds = self._exctract_features(X, return_preds=True)
        self._replace_model_head()

        X_pca = self.pca.transform(X_features)
        all_dists = np.array([cov.mahalanobis(X_pca) for cov in self.covariances]).T
        md = np.array([dist_i[pred_i] for pred_i, dist_i in zip(preds, all_dists)])

        md = pad_scores(md, np.asarray(y_pad).reshape(y_shape), y_pad)

        sum_inf_time = end - start
        eval_results["mahalanobis_distance"] = md.tolist()
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")

        log.info("**************Done.**********************")
        return eval_results
