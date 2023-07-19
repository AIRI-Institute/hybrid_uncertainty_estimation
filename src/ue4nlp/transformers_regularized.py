import numpy as np
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss
import torch.nn.functional as F

from transformers import BertModel, BertPreTrainedModel
from transformers import (
    ElectraForSequenceClassification,
    BertForSequenceClassification,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers import Trainer
from torch.autograd import Variable

from transformers.trainer_pt_utils import (
    nested_detach,
)
from transformers.file_utils import (
    is_sagemaker_mp_enabled,
)

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from transformers.trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


def entropy(x):
    return torch.sum(-x * torch.log(torch.clamp(x, 1e-8, 1)), axis=-1)


def conf(preds, probs, labels):
    conf_scores = torch.where(
        preds == labels,
        torch.max(probs, axis=-1).values,
        1 - torch.max(probs, axis=-1).values,
    )
    return conf_scores


class AsymmetricCELoss(nn.Module):
    def __init__(self, margin=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricCELoss, self).__init__()

        self.margin = margin
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.margin is not None and self.margin > 0:
            xs_neg = (xs_neg + self.margin).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        return -loss.sum()


def multilabel_loss(probs, labels, margin=0.05, gamma_neg=4, gamma_pos=1):
    loss_func = AsymmetricLossMultiLabel(
        gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=margin
    )
    labels_ohe = torch.nn.functional.one_hot(labels, num_classes=probs.shape[-1])
    loss = loss_func(probs, labels_ohe)

    return loss


def RAU_loss(probs, labels, unc_threshold=0.5, eps=1e-6):
    preds = torch.argmax(probs, axis=-1)
    conf_scores = conf(preds, probs, labels)
    uncertainty = entropy(probs)
    n_C = conf_scores * (1 - torch.tan(uncertainty))
    n_U = conf_scores * (torch.tan(uncertainty))

    n_AC = torch.where(
        (preds == labels) & (uncertainty <= unc_threshold),
        n_C,
        torch.tensor(0.0).to(labels.device),
    ).sum()
    n_AU = torch.where(
        (preds == labels) & (uncertainty > unc_threshold),
        n_U,
        torch.tensor(0.0).to(labels.device),
    ).sum()
    n_IC = torch.where(
        (preds != labels) & (uncertainty <= unc_threshold),
        n_C,
        torch.tensor(0.0).to(labels.device),
    ).sum()
    n_IU = torch.where(
        (preds != labels) & (uncertainty > unc_threshold),
        n_U,
        torch.tensor(0.0).to(labels.device),
    ).sum()
    loss = torch.log(1 + n_AU / (n_AC + n_AU + eps) + n_IC / (n_IC + n_IU + eps))
    return loss


def multiclass_metric_loss_fast(
    represent, target, margin=10, class_num=2, start_idx=1, per_class_norm=False
):
    target_list = target.data.tolist()
    dim = represent.data.shape[1]
    indices = []
    for class_idx in range(start_idx, class_num + start_idx):
        indice_i = [i for i, x in enumerate(target_list) if x == class_idx]
        indices.append(indice_i)

    loss_intra = torch.FloatTensor([0]).to(represent.device)
    num_intra = 0
    loss_inter = torch.FloatTensor([0]).to(represent.device)
    num_inter = 0
    for i in range(class_num):
        curr_repr = represent[indices[i]]
        s_k = len(indices[i])
        triangle_matrix = torch.triu(
            (curr_repr.unsqueeze(1) - curr_repr).norm(2, dim=-1)
        )
        buf_loss = torch.sum(1 / dim * (triangle_matrix**2))
        if per_class_norm:
            loss_intra += buf_loss / np.max([(s_k**2 - s_k), 1]) * 2
        else:
            loss_intra += buf_loss
            num_intra += (curr_repr.shape[0] ** 2 - curr_repr.shape[0]) / 2
        for j in range(i + 1, class_num):
            repr_j = represent[indices[j]]
            s_q = len(indices[j])
            matrix = (curr_repr.unsqueeze(1) - repr_j).norm(2, dim=-1)
            inter_buf_loss = torch.sum(
                torch.clamp(margin - 1 / dim * (matrix**2), min=0)
            )
            if per_class_norm:
                loss_inter += inter_buf_loss / np.max([(s_k * s_q), 1])
            else:
                loss_inter += inter_buf_loss
                num_inter += repr_j.shape[0] * curr_repr.shape[0]
    if num_intra > 0 and not (per_class_norm):
        loss_intra = loss_intra / num_intra
    if num_inter > 0 and not (per_class_norm):
        loss_inter = loss_inter / num_inter
    return loss_intra, loss_inter


def multiclass_metric_loss_fast_optimized(
    represent, target, margin=10, class_num=2, start_idx=1, per_class_norm=False
):
    target_list = target.data.tolist()
    dim = represent.data.shape[1]

    indices = []
    for class_idx in range(1, class_num + 1):
        indice_i = [i for i, x in enumerate(target_list) if x == class_idx]
        indices.append(indice_i)
    loss_intra = torch.FloatTensor([0]).to(represent.device)
    num_intra = 0
    loss_inter = torch.FloatTensor([0]).to(represent.device)
    num_inter = 0

    cls_repr = {}
    for i in range(class_num):
        indices_i = indices[i]
        curr_repr = represent[indices_i]
        if len(curr_repr) > 0:
            cls_repr[i] = curr_repr
            s_k = len(indices_i)
            triangle_matrix = torch.triu(
                (curr_repr.unsqueeze(1) - curr_repr).norm(2, dim=-1)
            )
            if per_class_norm:
                loss_intra += (
                    torch.sum(1 / dim * (triangle_matrix**2))
                    / np.max([(s_k**2 - s_k), 1])
                    * 2
                )
            else:
                loss_intra += torch.sum(1 / dim * (triangle_matrix**2))
                num_intra += (curr_repr.shape[0] ** 2 - curr_repr.shape[0]) / 2

    batch_labels = list(cls_repr.keys())
    bs = represent.shape[0]
    for n, j in enumerate(batch_labels):
        curr_repr = cls_repr[j]
        s_k = len(indices[j])
        matrices = torch.zeros(len(batch_labels), bs)
        inter_buf_loss = 0
        for l, k in enumerate(batch_labels[n + 1 :]):
            s_q = len(indices[k])
            matrix = (curr_repr.unsqueeze(1) - cls_repr[k]).norm(2, dim=-1).flatten()
            if per_class_norm:
                loss_inter += torch.sum(
                    torch.clamp(margin - 1 / dim * (matrix**2), min=0)
                ) / np.max([(s_k * s_q), 1])
            else:
                loss_inter += torch.sum(
                    torch.clamp(margin - 1 / dim * (matrix**2), min=0)
                )
                num_inter += cls_repr[k].shape[0] * curr_repr.shape[0]

    if num_intra > 0 and not (per_class_norm):
        loss_intra = loss_intra / num_intra
    if num_inter > 0 and not (per_class_norm):
        loss_inter = loss_inter / num_inter

    return loss_intra, loss_inter


def compute_loss_cer(logits, labels, loss, lamb, unpad=False):
    """Computes regularization term for loss with CER"""
    # here correctness is always 0 or 1
    if unpad:
        # NER case
        logits = logits[torch.nonzero(labels != -100, as_tuple=True)]
        labels = labels[torch.nonzero(labels != -100, as_tuple=True)]
    # suppose that -1 will works for ner and cls
    confidence, prediction = torch.softmax(logits, dim=-1).max(dim=-1)
    correctness = prediction == labels
    correct_confidence = torch.masked_select(confidence, correctness)
    wrong_confidence = torch.masked_select(confidence, ~correctness)
    regularizer = 0
    if unpad:
        # speed up for NER
        regularizer = torch.sum(
            torch.clamp(wrong_confidence.unsqueeze(1) - correct_confidence, min=0) ** 2
        )
    else:
        for cc in correct_confidence:
            for wc in wrong_confidence:
                regularizer += torch.clamp(wc - cc, min=0) ** 2
    loss += lamb * regularizer
    return loss


def compute_loss_metric(
    hiddens, labels, loss, num_labels, margin, lamb_intra, lamb, unpad=False
):
    """Computes regularization term for loss with Metric loss"""
    if unpad:
        hiddens = hiddens[torch.nonzero(labels != -100, as_tuple=True)]
        labels = labels[torch.nonzero(labels != -100, as_tuple=True)]
    class_num = num_labels
    start_idx = 0 if class_num == 2 else 1
    # TODO: define represent, target and margin
    # Get only sentence representaions
    (
        loss_intra,
        loss_inter,
    ) = multiclass_metric_loss_fast_optimized(  # multiclass_metric_loss_fast(
        hiddens,
        labels,
        margin=margin,
        class_num=class_num,
        start_idx=start_idx,
    )
    loss_metric = lamb_intra * loss_intra[0] + lamb * loss_inter[0]
    loss += loss_metric
    return loss


class AsymmetricLossSingleLabel(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=4, eps: float = 0.1, reduction="mean"):
        super(AsymmetricLossSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []  # prevent gpu repeated memory allocation
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target, reduction=None):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (1-hot vector)
        """

        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.nn.functional.one_hot(
            target, num_classes=inputs.shape[-1]
        )  # torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos

        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(
                self.eps / num_classes
            )

        # loss calculation
        loss = -self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class SelectiveLoss(torch.nn.Module):
    def __init__(self, coverage: float, lm: float = 32.0):
        """
        Args:
            loss_func: base loss function. the shape of loss_func(x, target) shoud be (B).
                       e.g.) torch.nn.CrossEntropyLoss(reduction=none) : classification
            coverage: target coverage.
            lm: Lagrange multiplier for coverage constraint. original experiment's value is 32.
        """
        super(SelectiveLoss, self).__init__()
        assert 0.0 < coverage <= 1.0
        assert 0.0 < lm

        self.coverage = coverage
        self.lm = lm
        self.base_loss = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, prediction_out, selection_out, target, threshold=0.5):
        """
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
        """
        # compute emprical coverage (=phi^)
        emprical_coverage = selection_out.mean()

        # compute emprical risk (=r^)
        emprical_risk = (
            self.base_loss(prediction_out, target) * selection_out.view(-1)
        ).mean()
        emprical_risk = emprical_risk / emprical_coverage

        # compute penulty (=psi)
        coverage = torch.tensor(
            self.coverage, dtype=torch.float32, requires_grad=True, device="cuda"
        )
        penalty = (
            torch.max(
                coverage - emprical_coverage,
                torch.tensor(
                    0.0, dtype=torch.float32, requires_grad=True, device="cuda"
                ),
            )
            ** 2
        )
        penalty *= self.lm

        selective_loss = emprical_risk + penalty

        # loss information dict
        loss_dict = {}
        loss_dict["emprical_coverage"] = emprical_coverage.detach().cpu().item()
        loss_dict["emprical_risk"] = emprical_risk.detach().cpu().item()
        loss_dict["penalty"] = penalty.detach().cpu().item()

        return selective_loss, loss_dict


class SelectiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = getattr(kwargs["args"], "task", "cls")
        self.reg_type = getattr(kwargs["args"], "reg_type", "reg-curr")
        self.lamb = getattr(kwargs["args"], "lamb", 0.01)
        self.margin = getattr(kwargs["args"], "margin", 0.5)
        self.lamb_intra = getattr(kwargs["args"], "lamb_intra", 0.5)
        self.unc_threshold = getattr(kwargs["args"], "unc_threshold", 0.5)

        self.gamma_pos = getattr(kwargs["args"], "gamma_pos", 1)
        self.gamma_neg = getattr(kwargs["args"], "gamma_neg", 4)

        self.coverage = getattr(kwargs["args"], "coverage", 0.9)
        self.lm = getattr(kwargs["args"], "lm", 32)
        self.alpha = getattr(kwargs["args"], "alpha", 0.1)

        self.use_amp = False
        if self.task == "cls":
            self.unpad = False
        else:
            self.unpad = True

    def post_init(self, reg_type, lamb, margin, lamb_intra, unc_threshold):
        """Add regularization params"""
        self.reg_type = reg_type
        self.lamb = lamb
        self.margin = margin
        self.lamb_intra = lamb_intra
        self.unc_threshold = unc_threshold

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        output_hidden_states = True if self.reg_type == "metric" else False
        outputs = model(**inputs, output_hidden_states=output_hidden_states)
        if self.reg_type == "selectivenet":
            logits = outputs.logits[:, : model.config.num_labels]
            selective = outputs.logits[
                :, model.config.num_labels : -model.config.num_labels
            ]
            cls_logits = outputs.logits[:, -model.config.num_labels :]
        else:
            logits = outputs.logits if self.task == "cls" else outputs[0]
        if self.reg_type == "metric":
            hiddens = (
                outputs.hidden_states[-1][:, 0, :]
                if self.task == "cls"
                else outputs[1][-1]
            )
            if self.task == "cls":
                del outputs
                torch.cuda.empty_cache()
                outputs = logits
        if model.config.num_labels == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        if self.reg_type == "raw":
            pass
        elif self.reg_type == "reg-curr":
            loss = compute_loss_cer(logits, labels, loss, self.lamb, unpad=self.unpad)
        elif self.reg_type == "metric":
            loss = compute_loss_metric(
                hiddens,
                labels,
                loss,
                model.config.num_labels,
                self.margin,
                self.lamb_intra,
                self.lamb,
                unpad=self.unpad,
            )
            if self.task == "ner":
                # we don't need hiddens anymore
                outputs = outputs[0]
        elif self.reg_type == "rau":
            loss += self.lamb * RAU_loss(
                torch.softmax(logits, dim=1), labels, self.unc_threshold
            )
        elif self.reg_type == "aml":
            loss = multilabel_loss(
                logits,
                labels.view(-1),
                margin=self.margin,
                gamma_pos=self.gamma_pos,
                gamma_neg=self.gamma_neg,
            )
        elif self.reg_type == "asymmetric":
            loss = AsymmetricLossSingleLabel(
                gamma_pos=self.gamma_pos, gamma_neg=self.gamma_neg
            )(logits, labels.view(-1))
        elif self.reg_type == "ceps":
            loss_fct = AsymmetricCELoss(margin=self.margin)
            labels_ohe = torch.nn.functional.one_hot(
                labels.view(-1), num_classes=logits.shape[-1]
            )
            loss = loss_fct(logits, labels_ohe)
        elif self.reg_type == "selectivenet":
            loss_func = SelectiveLoss(coverage=self.coverage, lm=self.lm)
            selective_loss, selective_loss_dict = loss_func(
                cls_logits, selective, labels
            )
            loss = (1 - self.alpha) * loss + self.alpha * selective_loss
        else:
            raise NotImplementedError()

        if self.reg_type == "selectivenet":
            outputs.selective = selective

        if isinstance(outputs, tuple):
            return (loss,) + outputs if return_outputs else loss
        else:
            return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )
                    loss = loss.mean().detach()
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        # Changed from original code - there was outputs[1:] for some reason
                        logits = outputs
                else:
                    loss = None
                    if self.use_amp:
                        with autocast():
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
