from ue4nlp.ue_estimator_mahalanobis import UeEstimatorMahalanobis
from ue4nlp.ue_estimator_rde import UeEstimatorRDE
from ue4nlp.ue_estimator_hybrid import UeEstimatorHybrid
from ue4nlp.ue_estimator_ddu import UeEstimatorDDU
import numpy as np

import logging

log = logging.getLogger(__name__)


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
    elif ue_args.ue_type == "rde":
        return UeEstimatorRDE(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "hybrid":
        return UeEstimatorHybrid(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "ddu":
        return UeEstimatorDDU(model, ue_args, config, train_dataset)
    else:
        raise ValueError()
