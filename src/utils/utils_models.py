from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from torch.nn.utils import spectral_norm
import torch
from utils.classification_models import (
    create_bert,
    create_xlnet,
    create_deberta,
    create_electra,
    create_roberta,
    create_distilbert,
    create_distilroberta,
    create_fairlib_bert,
    create_fairlib_mlp,
    build_model,
)

import logging

log = logging.getLogger(__name__)


def create_tokenizer(model_args, config):
    fairlib_args = config.get("fairlib", None)
    base_model_name = (
        model_args.model_name_or_path
        if fairlib_args is None
        else fairlib_args.model_name
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        cache_dir=config.cache_dir,
    )
    return tokenizer


def create_model(num_labels, model_args, data_args, ue_args, config):
    fairlib_args = config.get("fairlib", None)
    base_model_name = (
        model_args.model_name_or_path
        if fairlib_args is None
        else fairlib_args.model_name
    )
    model_config = AutoConfig.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=config.cache_dir,
    )

    if (
        data_args.task_name == "moji_preproc"
        or "fairlib_mlp" in model_args.model_name_or_path
        or "fairlib_fixed" in model_args.model_name_or_path
    ):
        # for this dataset we already have preprocessed embeddings
        # so we will use empty tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir=config.cache_dir,
        )
        tokenizer.__call__ = lambda instances, *args: instances
        tokenizer._call_one = lambda text, text_pair, *args, **kwargs: text
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir=config.cache_dir,
        )

    use_sngp = ue_args.ue_type == "sngp"
    use_duq = ue_args.ue_type == "duq"
    use_spectralnorm = "use_spectralnorm" in ue_args.keys() and ue_args.use_spectralnorm
    use_mixup = "mixup" in config.keys() and config.mixup.use_mixup
    use_selective = "use_selective" in ue_args.keys() and ue_args.use_selective
    model_path_or_name = model_args.model_name_or_path

    models_constructors = {
        "fairlib_bert": create_fairlib_bert,
        "fairlib_mlp": create_fairlib_mlp,
        "fairlib_fixed": create_fairlib_mlp,
        "electra": create_electra,
        "roberta": create_roberta,
        "distilroberta": create_distilroberta,
        "deberta": create_deberta,
        "distilbert": create_distilbert,
        "xlnet": create_xlnet,
        "bert": create_bert,
    }
    for key, value in models_constructors.items():
        if key in model_path_or_name:
            return (
                models_constructors[key](
                    model_config,
                    tokenizer,
                    use_sngp,
                    use_duq,
                    use_spectralnorm,
                    use_mixup,
                    use_selective,
                    ue_args,
                    model_path_or_name,
                    config,
                ),
                tokenizer,
            )
    raise ValueError(f"Cannot find model with this name or path: {model_path_or_name}")


def create_model_ner(num_labels, model_args, data_args, ue_args, config):
    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=config.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=config.cache_dir,
        use_fast=True,
        add_prefix_space=True,
    )

    use_spectralnorm = "use_spectralnorm" in ue_args.keys() and ue_args.use_spectralnorm
    use_mixup = "mixup" in config.keys() and config.mixup.use_mixup
    use_sngp = ue_args.ue_type == "sngp"
    use_selective = "use_selective" in ue_args.keys() and ue_args.use_selective
    model_path_or_name = model_args.model_name_or_path

    models_constructors = {
        "electra": create_electra_ner,
        "deberta": create_deberta_ner,
        "distilbert": create_distilbert_ner,
    }
    for key, value in models_constructors.items():
        if key in model_path_or_name:
            return (
                models_constructors[key](
                    model_config,
                    tokenizer,
                    use_sngp,
                    use_spectralnorm,
                    use_mixup,
                    use_selective,
                    ue_args,
                    model_path_or_name,
                    config,
                ),
                tokenizer,
            )
    raise ValueError(f"Cannot find model with this name or path: {model_name_or_path}")
    return model, tokenizer
