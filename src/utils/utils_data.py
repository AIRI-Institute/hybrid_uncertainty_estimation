import wget
import gzip
import os
import json
import numpy as np
import collections
import shutil
from datasets import Dataset, DatasetDict, load_dataset, load_metric, load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
import pytreebank
from datasets.arrow_dataset import Dataset as ArrowDataset
from torch.utils.data import Subset
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import reduce

import logging

log = logging.getLogger(__name__)

try:
    from toxigen import label_annotations
except:
    log.info("There is no toxigen module, will use version from main!")

    def label_annotations(annotated):
        # Annotations should be the annotated dataset
        label = ((annotated.toxicity_ai + annotated.toxicity_human) > 5.5).astype(int)
        labeled_annotations = pd.DataFrame()
        labeled_annotations["text"] = [i for i in annotated.text.tolist()]
        labeled_annotations["label"] = label
        return labeled_annotations


glue_datasets = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    # new datasets not from GLUE benchmark
    "20newsgroups": ("text", None),
    "amazon": ("text", None),
    "sst5": ("text", None),
    "twitter_hso": ("text", None),
    "imdb": ("text", None),
    "trec": ("text", None),
    "wmt16": ("text", None),
    "jigsaw_toxic": ("text", None),
    "paradetox": ("text", None),
    "toxigen": ("text", None),
    "dynahate": ("text", None),
    "implicithate": ("text", None),
    "sbf": ("text", None),
    "banking77": ("text", None),
    "tweet_eval": ("text", None),
    "amazon_massive": ("text", None),
    "bios": ("hard_text", "bert_avg_SE"),
    "rob_gender": ("x", "bert_avg_SE"),
    "rob_area": ("x", "bert_avg_SE"),
    "rob_ood": ("x", "bert_avg_SE"),
    "sepsis_ethnicity": ("x", "bert_avg_SE"),
    "sepsis_ood": ("x", "bert_avg_SE"),
    "moji_preproc": ("input_ids", None),
    "moji_raw": ("text", None),
    "trustpilot": ("text", "avg_embedding"),
    "jigsaw_race": ("comment_text", "bert_avg_SE"),
}


def load_data(config):
    if config.data.task_name == "20newsgroups":
        datasets = load_20newsgroups(config)
    elif config.data.task_name == "amazon":
        datasets = load_amazon_5core(config)
    elif config.data.task_name == "sst5":
        datasets = load_sst5(config)
    elif config.data.task_name == "twitter_hso":
        datasets = load_twitter_hso(config)
    elif config.data.task_name == "ag_news":
        datasets = load_ag_news(config)
    elif config.data.task_name == "jigsaw_toxic":
        datasets = load_jigsaw_toxic(config)
    elif config.data.task_name == "paradetox":
        datasets = load_paradetox(config)
    elif config.data.task_name == "toxigen":
        datasets = load_toxigen(config)
    elif config.data.task_name == "dynahate":
        datasets = load_dynahate(config)
    elif config.data.task_name == "implicithate":
        datasets = load_implicithate(config)
    elif config.data.task_name == "banking77":
        datasets = load_banking77(config)
    elif config.data.task_name == "tweet_eval":
        datasets = load_tweet_eval(config)
    elif config.data.task_name == "amazon_massive":
        datasets = load_amazon_massive(config)
    elif config.data.task_name == "sbf":
        datasets = load_sbf(config)
    elif "bios" in config.data.task_name:
        datasets = load_bios(config)
    elif "moji_preproc" in config.data.task_name:
        datasets = load_moji_preproc(config)
    elif "moji_raw" in config.data.task_name:
        datasets = load_moji_raw(config)
    elif "trustpilot" in config.data.task_name:
        datasets = load_trustpilot(config)
    elif "jigsaw_race" in config.data.task_name:
        datasets = load_jigsaw_race(config)
    elif "rob_gender" in config.data.task_name:
        datasets = load_rob(config)
    elif "rob_area" in config.data.task_name:
        datasets = load_rob(config, protected_label="area")
    elif "sepsis_ethnicity" in config.data.task_name:
        datasets = load_sepsis(config, protected_label="ethnicity")
    elif config.data.task_name in glue_datasets:
        datasets = load_dataset(
            "glue", config.data.task_name, cache_dir=config.cache_dir
        )
    else:
        raise ValueError(f"Cannot load dataset with this name: {config.data.task_name}")
    if config.data.get("balance_classes", False):
        for key in datasets.keys():
            labels, counts = np.unique(datasets[key]["label"], return_counts=True)
            sub_len = counts.min()
            sub_idx = []
            for label in labels:
                label_idx = np.where(np.array(datasets[key]["label"]) == label)[0]
                sub_idx += list(np.random.choice(label_idx, sub_len, replace=False))
            datasets[key] = datasets[key].select(sub_idx)
    return datasets


def load_data_adversarial(config):
    if config.data.task_name == "imdb":
        datasets = load_imdb_adversarial(config)
    elif config.data.task_name == "ag_news":
        datasets = load_ag_news_adversarial(config)
    elif config.data.task_name == "yelp":
        raise NotImplementedError(f"Not implemented for {config.data.task_name}")
    elif config.data.task_name == "sst2":
        raise NotImplementedError(f"Not implemented for {config.data.task_name}")
    else:
        raise ValueError(f"Cannot load dataset with this name: {config.data.task_name}")
    return datasets


def make_data_similarity(dataset):
    train = dataset["train"].to_pandas()
    test = dataset["validation"].to_pandas()
    train["label"] = 0
    test["label"] = 1
    data = pd.concat([train, test])

    train, test = train_test_split(
        data, stratify=data.label.values, test_size=0.1, random_state=42
    )

    datasets = DatasetDict(
        {
            "train": Dataset.from_pandas(train.reset_index(drop=True)),
            "validation": Dataset.from_pandas(test.reset_index(drop=True)),
        }
    )

    return datasets


def load_ag_news(config):
    dataset = load_dataset("ag_news", cache_dir=config.cache_dir)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    datasets = DatasetDict({"train": train_dataset, "validation": eval_dataset})

    return datasets


def load_banking77(config):
    dataset = load_dataset("banking77", cache_dir=config.cache_dir)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    datasets = DatasetDict({"train": train_dataset, "validation": eval_dataset})

    return datasets


def load_amazon_massive(config):
    dataset = load_dataset("AmazonScience/massive", cache_dir=config.cache_dir)

    train_en = np.array(dataset["train"]["locale"]) == "en-US"
    train_dataset = Dataset.from_dict(
        {
            "text": np.array(dataset["train"]["utt"])[train_en],
            "label": np.array(dataset["train"]["intent"])[train_en],
        }
    )

    test_en = np.array(dataset["test"]["locale"]) == "en-US"
    eval_dataset = Dataset.from_dict(
        {
            "text": np.array(dataset["test"]["utt"])[test_en],
            "label": np.array(dataset["test"]["intent"])[test_en],
        }
    )

    datasets = DatasetDict({"train": train_dataset, "validation": eval_dataset})

    return datasets


def load_tweet_eval(config):
    dataset = load_dataset("tweet_eval", "emoji", cache_dir=config.cache_dir)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    datasets = DatasetDict({"train": train_dataset, "validation": eval_dataset})

    return datasets


def load_ag_news_adversarial(config):
    dataset = load_dataset("ag_news", cache_dir=config.cache_dir)

    model_name = config.model.model_name_or_path.split("/")[-1].split("-")[0]
    adversarial_data_path = f"{config.data.adversarial_data_path}/ag-news_{config.data.attack}_{model_name}.csv"
    adversarial_dataset = pd.read_csv(adversarial_data_path, index_col=0)

    original_samples = adversarial_dataset.original_text.values
    adversarial_samples = adversarial_dataset.adversarial_text.values

    # Concatenate all original samples and their predictions
    x = np.concatenate((original_samples, adversarial_samples))
    y = np.concatenate(
        (np.zeros(len(original_samples)), np.ones(len(adversarial_samples)))
    )

    train_dataset = dataset["train"]
    eval_dataset = Dataset.from_dict({"text": x, "label": y})

    datasets = DatasetDict({"train": train_dataset, "validation": eval_dataset})

    return datasets


def load_imdb_adversarial(config):
    dataset = load_dataset(
        "imdb", ignore_verifications=True, cache_dir=config.cache_dir
    )

    model_name = config.model.model_name_or_path.split("/")[-1].split("-")[0]
    adversarial_data_path = f"{config.data.adversarial_data_path}/imdb_{config.data.attack}_{model_name}.csv"
    adversarial_dataset = pd.read_csv(adversarial_data_path, index_col=0)

    original_samples = adversarial_dataset.original_text.values
    adversarial_samples = adversarial_dataset.adversarial_text.values

    # Concatenate all original samples and their predictions
    x = np.concatenate((original_samples, adversarial_samples))
    y = np.concatenate(
        (np.zeros(len(original_samples)), np.ones(len(adversarial_samples)))
    )

    train_dataset = dataset["train"]
    eval_dataset = Dataset.from_dict({"text": x, "label": y})

    datasets = DatasetDict({"train": train_dataset, "validation": eval_dataset})

    return datasets


def load_jigsaw_toxic(config):
    train = pd.read_csv(config.data.data_path)
    train["label"] = (train[train.columns[2:]].sum(1) > 0) * 1
    train_sort = train.sort_values("label")

    X_train, X_test, y_train, y_test = train_test_split(
        train_sort.comment_text.values,
        train_sort.label.values,
        stratify=train_sort.label.values,
        test_size=0.1,
        random_state=42,
    )

    datasets = DatasetDict(
        {
            "train": Dataset.from_dict({"text": X_train, "label": y_train}),
            "validation": Dataset.from_dict({"text": X_test, "label": y_test}),
        }
    )
    return datasets


def load_paradetox(config):
    dataset = load_dataset("SkolkovoInstitute/paradetox", cache_dir=config.cache_dir)
    train = dataset["train"]
    toxic = train["en_toxic_comment"]
    neutral = train["en_neutral_comment"]

    text = toxic + neutral
    labels = [1] * len(toxic) + [0] * len(neutral)

    X_train, X_test, y_train, y_test = train_test_split(
        text, labels, stratify=labels, test_size=0.1, random_state=42
    )

    datasets = DatasetDict(
        {
            "train": Dataset.from_dict({"text": X_train, "label": y_train}),
            "validation": Dataset.from_dict({"text": X_test, "label": y_test}),
        }
    )
    return datasets


def load_toxigen(config):
    auth_token = "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"
    data = load_dataset(
        "skg/toxigen-data",
        name="annotated",
        use_auth_token=auth_token,
        cache_dir=config.cache_dir,
    )

    train = pd.DataFrame(data["train"])
    test = pd.DataFrame(data["test"])

    train = label_annotations(train)
    test = label_annotations(test)

    datasets = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"text": train.text.values, "label": train.label.values}
            ),
            "validation": Dataset.from_dict(
                {"text": test.text.values, "label": test.label.values}
            ),
        }
    )
    return datasets


def load_dynahate(config):
    data = load_dataset("aps/dynahate", cache_dir=config.cache_dir)["train"]

    train = pd.DataFrame(
        data.select(np.arange(0, len(data))[np.array(data["split"]) == "train"])
    )
    test = pd.DataFrame(
        data.select(np.arange(0, len(data))[np.array(data["split"]) == "test"])
    )

    datasets = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"text": train.text.values, "label": train.label.values}
            ),
            "validation": Dataset.from_dict(
                {"text": test.text.values, "label": test.label.values}
            ),
        }
    )
    return datasets


def load_implicithate(config):
    data = pd.read_csv(config.data.data_path, sep="\t")

    data["label"] = 0
    data.loc[data["class"] == "implicit_hate", "label"] = 1
    data.loc[data["class"] == "explicit_hate", "label"] = 2

    X_train, X_test, y_train, y_test = train_test_split(
        data["post"].values,
        data["label"].values,
        stratify=data["label"].values,
        test_size=0.1,
        random_state=42,
    )

    datasets = DatasetDict(
        {
            "train": Dataset.from_dict({"text": X_train, "label": y_train}),
            "validation": Dataset.from_dict({"text": X_test, "label": y_test}),
        }
    )
    return datasets


def load_sbf(config):
    dataset = load_dataset("social_bias_frames", cache_dir=config.cache_dir)

    train = pd.DataFrame(dataset["train"])
    test = pd.DataFrame(dataset["test"])

    train.offensiveYN = train.offensiveYN.apply(
        lambda x: 1 * (float(x) > 0) if len(x) else 0
    )
    test.offensiveYN = test.offensiveYN.apply(
        lambda x: 1 * (float(x) > 0) if len(x) else 0
    )

    train = (
        train.groupby(["HITId", "post"])[["offensiveYN"]].mean() > 0.5
    ).reset_index()
    test = (test.groupby(["HITId", "post"])[["offensiveYN"]].mean() > 0.5).reset_index()

    datasets = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": train.post.astype(str).values,
                    "label": train.offensiveYN.astype(int).values,
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "text": test.post.astype(str).values,
                    "label": test.offensiveYN.astype(int).values,
                }
            ),
        }
    )
    return datasets


def load_ood_dataset(
    dataset_path, max_seq_length, tokenizer, cache_dir=None, config=None
):
    log.info("Load out-of-domain dataset.")
    datasets_ood = load_dataset(
        dataset_path, ignore_verifications=True, cache_dir=cache_dir
    )
    log.info("Done with loading the dataset.")

    log.info("Preprocessing the dataset...")
    sentence1_key, sentence2_key = ("text", None)

    f_preprocess = lambda examples: preprocess_function(
        None, sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
    )

    datasets_ood = datasets_ood.map(
        f_preprocess,
        batched=True,
        load_from_cache_file=True,
    )

    ood_dataset = datasets_ood[config.ue.dropout.ood_sampling.subset].select(
        list(range(config.ue.dropout.ood_sampling.number_of_samples))
    )
    log.info("Done with preprocessing the dataset.")

    return ood_dataset


def load_ood_dataset_ner(
    dataset_path, data_args, tokenizer, cache_dir=None, config=None
):
    log.info("Load out-of-domain dataset.")
    datasets_ood = load_dataset(
        dataset_path, ignore_verifications=True, cache_dir=cache_dir
    )
    log.info("Done with loading the dataset.")

    log.info("Preprocessing the dataset...")

    text_column_name, label_column_name = "tokens", "ner_tags"
    label_to_id = {0: 0}
    f_preprocess = lambda examples: tokenize_and_align_labels(
        tokenizer,
        examples,
        text_column_name,
        label_column_name,
        data_args=data_args,
        label_to_id=label_to_id,
    )

    datasets_ood = datasets_ood.map(
        f_preprocess,
        batched=True,
        load_from_cache_file=True,
    )

    ood_dataset = datasets_ood[config.ue.dropout.ood_sampling.subset].select(
        list(range(config.ue.dropout.ood_sampling.number_of_samples))
    )
    ood_dataset = ood_dataset.remove_columns(["text", "label"])
    log.info("Done with preprocessing the dataset.")

    # for el in ood_dataset:
    #    print(len(el['ner_tags']), len(el['tokens']))
    # Have to drop labels col, otherwise it will be used by data_collator instead of ner_tags
    # ood_dataset["label"] = ood_dataset["ner_tags"]# ood_dataset.remove_columns("label")
    # ood_dataset = ood_dataset.remove_columns("label")
    return ood_dataset


def split_dataset(dataset, train_size=0.9, shuffle=True, seed=42):
    if isinstance(dataset, ArrowDataset):
        data = dataset.train_test_split(
            train_size=train_size, shuffle=shuffle, seed=seed
        )
        train_data, eval_data = data["train"], data["test"]
    else:
        train_idx, eval_idx = train_test_split(
            range(len(dataset)), shuffle=shuffle, random_state=seed
        )
        train_data = Subset(dataset, train_idx)
        eval_data = Subset(dataset, eval_idx)

    return train_data, eval_data


def tokenize_and_align_labels(
    tokenizer,
    examples,
    text_column_name,
    label_column_name,
    data_args,
    label_to_id,
    padding="max_length",
):
    if text_column_name not in examples:
        examples[text_column_name] = [exp.split(" ") for exp in examples["text"]]
        examples[label_column_name] = [
            [0] * len(exp.split(" ")) for exp in examples["text"]
        ]

    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        max_length=data_args.max_seq_length,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(
                    label_to_id[label[word_idx]] if data_args.label_all_tokens else -100
                )

            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def preprocess_function(
    label_to_id, sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
):
    # Tokenize the texts
    result = preprocess_text_only(
        sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
    )

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [
            (label_to_id[l] if l != -1 else -1) for l in examples["label"]
        ]
    return result


def preprocess_text_only(
    sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(
        *args, padding="max_length", max_length=max_seq_length, truncation=True
    )
    return result


def load_amazon_5core(config):
    """Return closest version of Amazon Reviews Sports & Outdoors split from the paper
    Towards More Accurate Uncertainty Estimation In Text Classification.
    """
    texts, targets = [], []
    # get zipped dataset
    url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz"
    save_path = os.path.join(config.cache_dir, "amazon_5core.json.gz")
    # check if file already exists, load if not
    if not (os.path.isfile(save_path)):
        save_path = wget.download(url, out=save_path)
    # unzip it and extract data to arrays
    with gzip.open(save_path, "rb") as f:
        for line in f.readlines():
            data = json.loads(line)
            texts.append(data["reviewText"])
            targets.append(np.int64(data["overall"]))
    # to shift classes from 1-5 to 0-4
    targets = np.asarray(targets) - 1
    # split on train|val|test
    text_buf, text_eval, targ_buf, targ_eval = train_test_split(
        texts, targets, test_size=0.1, random_state=config.seed
    )
    text_train, text_val, targ_train, targ_val = train_test_split(
        text_buf, targ_buf, test_size=2.0 / 9.0, random_state=config.seed
    )
    amazon_train = {"label": targ_train, "text": text_train}
    amazon_eval = {"label": targ_eval, "text": text_eval}
    datasets = DatasetDict(
        {
            "train": Dataset.from_dict(amazon_train),
            "validation": Dataset.from_dict(amazon_eval),
        }
    )
    return datasets


def load_20newsgroups(config):
    newsgroups_train = fetch_20newsgroups(subset="train")
    newsgroups_train = {
        "label": newsgroups_train["target"],
        "text": newsgroups_train["data"],
    }
    newsgroups_eval = fetch_20newsgroups(subset="test")
    newsgroups_eval = {
        "label": newsgroups_eval["target"],
        "text": newsgroups_eval["data"],
    }
    datasets = DatasetDict(
        {
            "train": Dataset.from_dict(newsgroups_train),
            "validation": Dataset.from_dict(newsgroups_eval),
        }
    )
    return datasets


def load_sst5(config):
    dataset = pytreebank.load_sst()
    sst_datasets = {}
    for category in ["train", "test", "dev"]:
        df = {"text": [], "label": []}
        for item in dataset[category]:
            df["text"].append(item.to_labeled_lines()[0][1])
            df["label"].append(item.to_labeled_lines()[0][0])
        cat_name = category if category != "dev" else "validation"
        sst_datasets[cat_name] = Dataset.from_dict(df)
    dataset = DatasetDict(sst_datasets)
    return dataset


def load_mnli(config, matched: bool = True, annotator: int = -1):
    """Return MNLI dataset in different versions - matched/mismatched,
    with val labels from one annotator.
    Input:
    matched: bool, load matched or mismatched dev part
    annotator: int, annotator index. Should be in range 0-4, -1 means
    that we load mean label by all annotators for dev part
    """
    # get zipped dataset
    assert (
        annotator >= -1 and annotator <= 4
    ), "Annotator index should be int from -1 to 4"
    url = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
    save_path = os.path.join(config.cache_dir, "mnli_1_0.zip")
    if not (os.path.isfile(save_path)):
        print("File doesn't found")
        save_path = wget.download(url, out=save_path)
    print("Loaded")
    # after unpack folder
    if not (os.path.isdir(os.path.join(config.cache_dir, "multinli_1.0"))):
        print("Extracting archive")
        shutil.unpack_archive(save_path, config.cache_dir)
    train_path = os.path.join(config.cache_dir, "multinli_1.0/multinli_1.0_train.jsonl")
    if matched:
        dev_path = os.path.join(
            config.cache_dir, "multinli_1.0/multinli_1.0_dev_matched.jsonl"
        )
    else:
        dev_path = os.path.join(
            config.cache_dir, "multinli_1.0/multinli_1.0_dev_mismatched.jsonl"
        )

    def read_fields(data_path, annotator):
        data_texts1, data_texts2, data_targets = [], [], []
        target_key = "annotator_labels"
        with open(data_path, "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                data_texts1.append(data["sentence1"])
                data_texts2.append(data["sentence2"])
                if annotator == -1:
                    # get the most frequent label
                    data_targets.append(
                        collections.Counter(data[target_key]).most_common()[0][0]
                    )
                else:
                    data_targets.append(data[target_key][annotator])
        return data_texts1, data_texts2, data_targets

    # for train part set idx to 0, because there is only one label on train
    train_texts1, train_texts2, train_targets = read_fields(train_path, 0)
    dev_texts1, dev_texts2, dev_targets = read_fields(dev_path, annotator)
    # after encode targets as int classes
    target_encoder = LabelEncoder()
    train_targets = target_encoder.fit_transform(train_targets)
    dev_targets = target_encoder.transform(dev_targets)
    # and finally build dataset
    mnli_train = {
        "label": train_targets,
        "sentence1": train_texts1,
        "sentence2": train_texts2,
    }
    mnli_eval = {"label": dev_targets, "sentence1": dev_texts1, "sentence2": dev_texts2}
    datasets = DatasetDict(
        {
            "train": Dataset.from_dict(mnli_train),
            "validation": Dataset.from_dict(mnli_eval),
        }
    )
    return datasets


def load_twitter_hso(config):
    dataset = load_dataset("hate_speech_offensive", cache_dir=config.cache_dir)
    df = dataset["train"].to_pandas()
    annotators_count_cols = [
        "hate_speech_count",
        "offensive_language_count",
        "neither_count",
    ]

    # split by ambiguity (for test select most ambiguous part by annotators disagreement)
    df_test = df[df["count"] != df[annotators_count_cols].max(axis=1)].reset_index(
        drop=True
    )
    df_train = df[df["count"] == df[annotators_count_cols].max(axis=1)].reset_index(
        drop=True
    )

    train_dataset = {"label": df_train["class"], "text": df_train["tweet"]}

    eval_dataset = {"label": df_test["class"], "text": df_test["tweet"]}

    datasets = DatasetDict(
        {
            "train": Dataset.from_dict(train_dataset),
            "validation": Dataset.from_dict(eval_dataset),
        }
    )

    return datasets


def load_data_ood(
    dataset, config, data_args, dataset_type="plus", split="", tokenizer=None
):
    if dataset == "clinc_oos":
        # Load CLINC dataset. Types could be 'small', 'imbalanced', 'plus'. 'plus' type stands for CLINC-150, used in paper on Mahalonobis distance.
        log.info("Load dataset.")
        datasets = load_dataset(
            dataset, dataset_type, cache_dir=config.cache_dir
        )  # load_dataset("glue", config.data.task_name, cache_dir=config.cache_dir)
        log.info("Done with loading the dataset.")

        datasets = datasets.rename_column("intent", "label")
        datasets["train"] = datasets["train"].filter(lambda x: x["label"] != 42)

        def map_classes(examples):
            examples["label"] = (
                examples["label"] if (examples["label"] < 42) else examples["label"] - 1
            )
            return examples

        datasets["train"] = datasets["train"].map(
            map_classes,
            batched=False,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    elif dataset in ["rostd", "snips", "rostd_coarse"]:
        # Load ROSTD/ROSTD-Coarse/SNIPS dataset
        if not (config.data.get("data_dir", False)):
            raise ValueError(
                "For ROSTD or SNIPS dataset you need to set config.data.data_dir"
            )
        if split != "":
            if split == 0:
                config.data.data_dir += f"unsup_0.75_{split}/"
            else:
                config.data.data_dir = (
                    os.path.dirname(config.data.data_dir[:-1]) + f"/unsup_0.75_{split}/"
                )

        datasets = load_dataset(
            "csv",
            data_files={
                "train": config.data.data_dir + "OODRemovedtrain.tsv",
                "validation": config.data.data_dir + "eval.tsv",
                "test": config.data.data_dir + "test.tsv",
            },
            delimiter="\t",
            column_names=["label", "smth", "text"],
            index_col=False,
        )
        # Make labels dict with last class as OOD class
        labels = datasets["validation"].unique("label")
        labels.remove("outOfDomain")
        labels2id = {label: idx for idx, label in enumerate(labels)}
        labels2id["outOfDomain"] = len(labels2id)

        # TODO: encode labels
        def map_classes(examples):
            examples["label"] = labels2id[examples["label"]]
            return examples

        datasets = datasets.map(map_classes, batched=False)
    elif dataset in [
        "sst2",
        "20newsgroups",
        "amazon",
        "bios",
        "trustpilot",
        "moji_raw",
        "jigsaw_race",
        "rob_area",
        "rob_gender",
        "sepsis_ethnicity",
        "moji_preproc",
    ]:
        log.info(
            f"Loading {dataset} as ID dataset and {config.data.ood_data} as OOD dataset."
        )

        if config.data.task_name == "20newsgroups":
            id_datasets = load_20newsgroups(config)
        elif config.data.task_name == "amazon":
            id_datasets = load_amazon_5core(config)
        elif config.data.task_name == "sst2":
            id_datasets = load_dataset("glue", dataset, cache_dir=config.cache_dir)
        elif config.data.task_name == "bios":
            id_datasets = load_bios(config)
        elif config.data.task_name == "trustpilot":
            id_datasets = load_trustpilot(config)
        elif config.data.task_name == "moji_raw":
            id_datasets = load_moji_raw(config)
        elif config.data.task_name == "jigsaw_race":
            id_datasets = load_jigsaw_race(config)
        elif config.data.task_name == "rob_gender":
            id_datasets = load_rob(config)
        elif config.data.task_name == "rob_area":
            id_datasets = load_rob(config, protected_label="area")
        elif config.data.task_name == "sepsis_ethnicity":
            id_datasets = load_sepsis(config, protected_label="ethnicity")
        elif config.data.task_name == "moji_preproc":
            id_datasets = load_moji_preproc(config)
        else:
            raise ValueError(
                f"Cannot load dataset with this name: {config.data.task_name}"
            )

        if "idx" in id_datasets.column_names["train"]:
            id_datasets = id_datasets.remove_columns("idx")

        log.info("Done with loading the ID dataset.")

        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
        sentence2_key = (
            None
            if (
                config.data.task_name
                in [
                    "bios",
                    "trustpilot",
                    "jigsaw_race",
                    "rob_area",
                    "rob_gender",
                    "sepsis_ethnicity",
                ]
            )
            else sentence2_key
        )
        # omit padding if we want to load pretrained embeddings
        is_mlp = ("fixed" in config.model.model_name_or_path) or (
            "mlp" in config.model.model_name_or_path
        )
        if not is_mlp:
            padding = "max_length"
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
            f_preprocess = lambda examples: preprocess_function(
                None, sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
            )

            id_datasets = id_datasets.map(
                f_preprocess,
                batched=True,
                load_from_cache_file=False,  # not data_args.overwrite_cache,
            )

        train_dataset = id_datasets["train"]
        if config.data.task_name == "mnli":
            id_test_data = id_datasets["validation_mismatched"]
        elif config.data.task_name in glue_datasets + ["20newsgroups", "amazon"]:
            id_test_data = id_datasets["validation"]
        else:
            id_test_data = id_datasets["test"]

        log.info("Done with preprocessing the ID dataset.")

        if config.data.get("load_local", False):
            ood_datasets = load_from_disk(
                os.path.join(config.data.load_local, config.data.ood_data)
            )
        else:
            if config.data.ood_data == "20newsgroups":
                ood_datasets = load_20newsgroups(config)
            elif config.data.ood_data == "amazon":
                ood_datasets = load_amazon_5core(config)
            elif config.data.ood_data in glue_datasets:
                ood_datasets = load_dataset(
                    "glue", config.data.ood_data, cache_dir=config.cache_dir
                )
            elif config.data.ood_data in ["imdb", "trec"]:
                ood_datasets = load_dataset(
                    config.data.ood_data,
                    ignore_verifications=True,
                    cache_dir=config.cache_dir,
                )
            elif config.data.ood_data == "wmt16":
                ood_datasets = load_dataset(
                    config.data.ood_data, "de-en", cache_dir=config.cache_dir
                )
            elif config.data.ood_data == "sepsis_ood":
                ood_datasets = load_sepsis_ood(config)
            elif config.data.ood_data == "rob_ood":
                ood_datasets = load_rob_ood(config)
            else:
                raise ValueError(
                    f"Cannot load dataset with this name: {config.data.ood_data}"
                )

        if config.data.ood_data == "wmt16":
            ood_dataset = {
                "text": [
                    example["en"] for example in ood_datasets["test"]["translation"]
                ],
                "label": [0] * len(ood_datasets["test"]["translation"]),
            }
            ood_dataset = Dataset.from_dict(ood_dataset)
            ood_datasets = DatasetDict({"test": ood_dataset})

        log.info("Done with loading the OOD dataset.")

        sentence1_key, sentence2_key = task_to_keys[config.data.ood_data]
        sentence2_key = (
            None
            if (
                config.data.task_name
                in [
                    "bios",
                    "trustpilot",
                    "jigsaw_race",
                    "rob_area",
                    "rob_gender",
                    "sepsis_ethnicity",
                    "sepsis_ood",
                    "rob_ood",
                ]
            )
            else sentence2_key
        )

        padding = "max_length"
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        f_preprocess = lambda examples: preprocess_function(
            None, sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
        )

        ood_datasets = ood_datasets.map(
            f_preprocess,
            batched=True,
            load_from_cache_file=False,  # not data_args.overwrite_cache,
        )

        if config.data.ood_data == "mnli":
            ood_test_data = ood_datasets["validation_mismatched"]
        elif config.data.ood_data in glue_datasets + ["20newsgroups", "amazon"]:
            ood_test_data = ood_datasets["validation"]
        else:
            ood_test_data = ood_datasets["test"]

        log.info("Done with preprocessing the OOD dataset.")
        if not is_mlp:
            data_keys = ["input_ids", "token_type_ids", "attention_mask", "label"]
            new_train_dataset = {}
            new_test_dataset = {}
            for key in data_keys:
                new_train_dataset[key] = train_dataset[key]
                if key == "label":
                    new_test_dataset[key] = [0] * len(id_test_data["input_ids"]) + [
                        1
                    ] * len(ood_test_data["input_ids"])
                else:
                    new_test_dataset[key] = list(id_test_data[key]) + list(
                        ood_test_data[key]
                    )
        else:
            # in this case simple mix already pretrained embeddings
            # if we don't have one, extract it from bert-base-cased
            # now extract embeddings from ood_data
            if dataset == "moji_preproc":
                from utils.utils_encoding import encode_moji

                ood_text_key = task_to_keys[config.data.ood_data][0]
                text_data = list(ood_test_data[ood_text_key])
                moji_embeddings = encode_moji(text_data, config)
                ood_test_embeddings = list(moji_embeddings)
            else:
                from fairlib.datasets.utils.bert_encoding import BERT_encoder

                model_name = config.fairlib.get("model_name", "bert-base-cased")
                encoder = BERT_encoder(
                    config.training.per_device_train_batch_size, model_name
                )
                ood_text_key = task_to_keys[config.data.ood_data][0]
                text_data = list(ood_test_data[ood_text_key])
                avg_data, cls_data = encoder.encode(text_data)
                ood_test_embeddings = list(avg_data)
            data_keys = ["input_ids", "label"]
            new_train_dataset = {}
            new_test_dataset = {}
            for key in data_keys:
                new_train_dataset[key] = train_dataset[key]
                if key == "label":
                    new_test_dataset[key] = [0] * len(id_test_data["input_ids"]) + [
                        1
                    ] * len(ood_test_embeddings)
                else:
                    new_test_dataset["input_ids"] = list(
                        id_test_data["input_ids"]
                    ) + list(ood_test_embeddings)

        train_dataset = Dataset.from_dict(new_train_dataset)
        test_dataset = Dataset.from_dict(new_test_dataset)

        datasets = DatasetDict({"train": train_dataset, "test": test_dataset})

    else:
        raise ValueError(
            "Task name for OOD must be clinc_oos, rostd, rostd_coarse or snips"
        )
    return datasets


def subsample_balance_bios(config, split, X, y, protected_label):
    if config.fairlib.get("subsample_all", 1.0) < 1.0:
        # subsample split
        subsample_len = int(len(y) * config.fairlib.subsample_all)
        X = X[:subsample_len]
        y = y[:subsample_len]
        protected_label = protected_label[:subsample_len]
    if config.fairlib.get("balance_test", False) and split in [
        "test",
        "dev",
        "validation",
    ]:
        classes = np.unique(y)
        # rebalance test set
        overall_mask_bios = np.array([False] * len(y))
        for class_val in classes:
            class_ids = np.where(y == class_val, True, False)
            class_ids = np.arange(len(y))[class_ids]
            # find prot_attr distribution
            vals, distr = np.unique(protected_label[class_ids], return_counts=True)
            min_val, min_attr = np.min(distr), np.argmin(distr)
            min_ids = class_ids[np.where(protected_label[class_ids] == min_attr)[0]]
            max_ids = class_ids[
                np.where(protected_label[class_ids] != min_attr)[0][:min_val]
            ]
            # print(min_ids, class_ids)
            # print(vals,distr)
            # print(len(min_ids), len(max_ids))
            np.put(overall_mask_bios, min_ids, True)
            np.put(overall_mask_bios, max_ids, True)
        test_mask_ids = np.arange(len(overall_mask_bios))[overall_mask_bios]
        # print(np.sum(overall_mask_bios), len(y))
        X = list(np.asarray(X)[test_mask_ids])
        y = y[test_mask_ids]
        protected_label = protected_label[test_mask_ids]
    return X, y, protected_label


def subsample_balance_rob_sepsis(
    config, split, X, y, protected_label, dev_y, dev_protected_label
):
    if config.fairlib.get("subsample_all", 1.0) < 1.0:
        # subsample split
        subsample_len = int(len(y) * config.fairlib.subsample_all)
        X = X[:subsample_len]
        y = y[:subsample_len]
        protected_label = protected_label[:subsample_len]

    if config.fairlib.get("subsample_protected_labels", False):
        subsample_min_size = config.fairlib.get("subsample_min_size", 20)
        classes = np.unique(dev_y)
        large_attrs_cls = []
        for c in classes:
            value_counts = dev_protected_label[dev_y == c].value_counts()
            large_attrs_cls.append(
                value_counts[value_counts > subsample_min_size].index
            )
        large_attrs = reduce(np.intersect1d, large_attrs_cls)

        ids = np.isin(protected_label, large_attrs)

        X = list(np.asarray(X)[ids])
        y = y[ids].reset_index(drop=True)
        protected_label = protected_label[ids].reset_index(drop=True)

    if config.fairlib.get("balance_test", False) and split in [
        "test",
        "dev",
        "validation",
    ]:
        classes = np.unique(y)
        attrs = np.unique(protected_label)
        # rebalance test set
        overall_mask = np.array([False] * len(y))
        for class_val in classes:
            class_ids = np.where(y == class_val, True, False)
            class_ids = np.arange(len(y))[class_ids]
            # find prot_attr distribution
            vals, distr = np.unique(protected_label[class_ids], return_counts=True)
            min_val, min_attr = np.min(distr), np.argmin(distr)
            min_val = max(min_val, 20)
            min_ids = class_ids[np.where(protected_label[class_ids] == min_attr)[0]]
            np.put(overall_mask, min_ids, True)
            for attr in attrs:
                if attr == min_attr:
                    continue
                idx = protected_label[class_ids] == attr
                if min_val < idx.sum():
                    max_ids = class_ids[np.where(idx)[0][:min_val]]
                else:
                    max_ids = class_ids[np.where(idx)[0]]
                np.put(overall_mask, max_ids, True)
        test_mask_ids = np.arange(len(overall_mask))[overall_mask]
        X = list(np.asarray(X)[test_mask_ids])
        y = y[test_mask_ids]
        protected_label = protected_label[test_mask_ids]
    return X, y, protected_label


def load_bios(config):
    """Loads BIOS dataset from fairlib."""
    from fairlib.datasets import bios

    save_path = os.path.join(config.cache_dir, config.data.task_name)
    if config.data.get("task_dir", False):
        save_path = os.path.join(config.cache_dir, config.data.task_dir)
    log.info(save_path)
    batch_size = 16 if not config.get("fairlib", False) else config.fairlib.batch_size

    dataset_class = bios.init_data_class(
        dest_folder=save_path,
        batch_size=batch_size,
        model_name="bert-base-cased"
        if not config.get("fairlib", False)
        else config.fairlib.model_name,
    )
    splits = ["train", "dev", "test"]
    # check if we already have cached dataset
    # list of data files in folder
    files = [os.path.join(save_path, "{}.pickle".format(split)) for split in splits]
    files_df = [
        os.path.join(save_path, "bios_{}_df.pkl".format(split)) for split in splits
    ]
    files += files_df
    if not (all([os.path.isfile(file) for file in files])):
        dataset_class.prepare_data()
    # now load processed data from cache_dir and transform to Dataset
    datasets_dict = {}
    splits = ["train", "validation", "test"]
    for file, split in zip(files_df, splits):
        data = pd.read_pickle(file)
        if config.fairlib.get("subsample_classes", False):
            # use only selected columns from target class
            selected_columns = config.fairlib.subsample_classes.split("-")
            data = data[data["p"].isin(selected_columns)]
            # after reindex target labels
            data = data.reset_index(drop=True)
            old_targets = data["profession_class"].unique()
            old_targets.sort()
            map_new_targets = {
                key: value for key, value in zip(old_targets, range(len(old_targets)))
            }
            data["profession_class"].replace(map_new_targets, inplace=True)
        is_mlp = ("fixed" in config.model.model_name_or_path) or (
            "mlp" in config.model.model_name_or_path
        )
        prot_attr_name = (
            "gender_class"
            if not config.get("fairlib", False)
            else config.fairlib.prot_attr_name
        )
        prot_attr = data[prot_attr_name].astype(np.int)
        if not is_mlp:
            X, y, _ = subsample_balance_bios(
                config,
                split,
                data[task_to_keys["bios"][0]],
                data["profession_class"].astype(np.int),
                prot_attr,
            )
            datasets_dict[split] = Dataset.from_dict(
                {
                    task_to_keys["bios"][0]: X,
                    "label": y,
                }
            )
        else:
            X, y, _ = subsample_balance_bios(
                config,
                split,
                data[task_to_keys["bios"][1]],
                data["profession_class"].astype(np.int),
                prot_attr,
            )
            datasets_dict[split] = Dataset.from_dict(
                {
                    "input_ids": X,
                    "label": y,
                }
            )
    datasets = DatasetDict(datasets_dict)
    return datasets


def load_rob(config, protected_label="gender"):
    """Loads RoB dataset"""
    save_path = config.cache_dir
    splits = ["train", "dev", "test"]

    if "biobert" in config.fairlib.model_name:
        short_model_name = "biobert_"
    elif "scibert" in config.fairlib.model_name:
        short_model_name = "scibert_"
    else:
        short_model_name = ""

    if protected_label == "gender":
        files = [
            os.path.join(save_path, "{}rob_{}_df.pkl".format(short_model_name, split))
            for split in splits
        ]
        dev_data = pd.read_pickle(
            os.path.join(save_path, "{}rob_dev_df.pkl".format(short_model_name))
        )
    else:
        files = [
            os.path.join(
                save_path,
                "{}rob_{}_{}_df.pkl".format(short_model_name, protected_label, split),
            )
            for split in splits
        ]
        dev_data = pd.read_pickle(
            os.path.join(
                save_path,
                "{}rob_{}_dev_df.pkl".format(short_model_name, protected_label),
            )
        )

    dev_y = dev_data["y"].astype(np.float64)
    dev_protected_label = dev_data[f"{protected_label}_class"].astype(np.int)

    datasets_dict = {}
    splits = ["train", "validation", "test"]
    for file, split in zip(files, splits):
        data = pd.read_pickle(file)
        is_mlp = ("fixed" in config.model.model_name_or_path) or (
            "mlp" in config.model.model_name_or_path
        )
        prot_attr = data[f"{protected_label}_class"].astype(np.int)
        if not is_mlp:
            X, y, _ = subsample_balance_rob_sepsis(
                config,
                split,
                data[task_to_keys[config.data.task_name][0]],
                data["y"].astype(np.int),
                prot_attr,
                dev_y,
                dev_protected_label,
            )

            datasets_dict[split] = Dataset.from_dict(
                {
                    task_to_keys[config.data.task_name][0]: X,
                    "label": y,
                }
            )
        else:
            X, y, _ = subsample_balance_rob_sepsis(
                config,
                split,
                data[task_to_keys[config.data.task_name][1]],
                data["y"].astype(np.int),
                prot_attr,
                dev_y,
                dev_protected_label,
            )

            datasets_dict[split] = Dataset.from_dict(
                {
                    "input_ids": X,
                    "label": y,
                }
            )
    datasets = DatasetDict(datasets_dict)
    return datasets


def load_sepsis(config, protected_label="ethnicity"):
    """Loads Sepsis dataset"""
    save_path = config.cache_dir
    splits = ["train", "dev", "test"]
    files = [
        os.path.join(save_path, "sepsis_{}_{}_df.pkl".format(protected_label, split))
        for split in splits
    ]

    dev_data = pd.read_pickle(
        os.path.join(save_path, "sepsis_{}_dev_df.pkl".format(protected_label))
    )
    dev_y = dev_data["label"].astype(np.float64)
    dev_protected_label = dev_data[f"{protected_label}_class"].astype(np.int)

    datasets_dict = {}
    splits = ["train", "validation", "test"]
    for file, split in zip(files, splits):
        data = pd.read_pickle(file)
        is_mlp = ("fixed" in config.model.model_name_or_path) or (
            "mlp" in config.model.model_name_or_path
        )
        prot_attr = data[f"{protected_label}_class"].astype(np.int)
        if not is_mlp:
            X, y, _ = subsample_balance_rob_sepsis(
                config,
                split,
                data[task_to_keys[config.data.task_name][0]],
                data["label"].astype(np.int),
                prot_attr,
                dev_y,
                dev_protected_label,
            )

            datasets_dict[split] = Dataset.from_dict(
                {
                    task_to_keys[config.data.task_name][0]: X,
                    "label": y,
                }
            )
        else:
            X, y, _ = subsample_balance_rob_sepsis(
                config,
                split,
                data[task_to_keys[config.data.task_name][1]],
                data["label"].astype(np.int),
                prot_attr,
                dev_y,
                dev_protected_label,
            )

            datasets_dict[split] = Dataset.from_dict(
                {
                    "input_ids": X,
                    "label": y,
                }
            )
    datasets = DatasetDict(datasets_dict)
    return datasets


def load_sepsis_ood(config):
    """Loads Sepsis dataset"""
    save_path = config.cache_dir
    file = os.path.join(
        save_path, "sepsis_ood/{}.json".format(config.fairlib.ood_filename)
    )

    datasets_dict = {}
    data = pd.read_json(file)
    data.columns = [col.lower() for col in data.columns]

    is_mlp = ("fixed" in config.model.model_name_or_path) or (
        "mlp" in config.model.model_name_or_path
    )
    split = "test"

    if not is_mlp:
        datasets_dict[split] = Dataset.from_dict(
            {
                task_to_keys[config.data.task_name][0]: data["text"],
                "label": [1] * len(data),
            }
        )
    else:
        datasets_dict[split] = Dataset.from_dict(
            {
                "input_ids": data[task_to_keys[config.data.task_name][1]],
                "label": [1] * len(data),
            }
        )
    datasets = DatasetDict(datasets_dict)
    return datasets


def load_rob_ood(config):
    """Loads rob ood dataset"""
    save_path = config.cache_dir
    file = os.path.join(
        save_path, "risk-of-bias_ood/{}.json".format(config.fairlib.ood_filename)
    )

    datasets_dict = {}
    data = pd.read_json(file)
    data.columns = [col.lower() for col in data.columns]

    # Subsample to size of RoB test dataset
    data = data.sample(n=4254, random_state=42)

    is_mlp = ("fixed" in config.model.model_name_or_path) or (
        "mlp" in config.model.model_name_or_path
    )
    split = "test"

    if not is_mlp:
        datasets_dict[split] = Dataset.from_dict(
            {
                task_to_keys[config.data.task_name][0]: data["text"],
                "label": [1] * len(data),
            }
        )
    else:
        datasets_dict[split] = Dataset.from_dict(
            {
                "input_ids": data[task_to_keys[config.data.task_name][1]],
                "label": [1] * len(data),
            }
        )
    datasets = DatasetDict(datasets_dict)
    return datasets


def load_moji_preproc(config):
    """Loads Moji dataset from fairlib (preprocessed version for MLP)."""
    from fairlib.datasets import moji

    save_path = os.path.join(config.cache_dir, config.data.task_name)
    dataset_class = moji.init_data_class(
        dest_folder=save_path, batch_size=config.fairlib.batch_size
    )
    splits = ["train", "dev", "test"]
    # check if we already have cached dataset
    # list of data files in folder
    files = ["pos_pos.npy", "pos_neg.npy", "neg_pos.npy", "neg_neg.npy"]
    # for each split we have folder with files
    files_by_split = [
        os.path.join(save_path, split, file) for file in files for split in splits
    ]
    if not (all([os.path.isfile(file) for file in files_by_split])):
        dataset_class.prepare_data()
    # now load processed data from cache_dir and transform to Dataset
    # for more info look at the preprocessing script for Moji dataset in fairlib
    datasets_dict = {}
    splits_map = {"train": "train", "validation": "dev", "test": "test"}
    # calc percentage of used data for each file
    ratio = (
        lambda split: 0.8
        if (
            (split == "train" and not config.data.get("balanced", False))
            or config.data.get("unbalance_test", False)
        )
        else 0.5
    )
    p_aae = 0.5  # proportion of the AAE
    n = 100000
    n_1 = lambda split: int(n * p_aae * ratio(split))  # happy AAE
    n_2 = lambda split: int(n * (1 - p_aae) * (1 - ratio(split)))  # happy SAE
    n_3 = lambda split: int(n * p_aae * (1 - ratio(split)))  # unhappy AAE
    n_4 = lambda split: int(n * (1 - p_aae) * ratio(split))  # unhappy SAE
    if config.data.get("unbalance_test", False):
        # recalc n_i by real test and val sizes
        # mimic train distribution
        n_1 = lambda split: 2000 if split != "train" else int(n * p_aae * ratio(split))
        n_4 = (
            lambda split: 2000
            if split != "train"
            else int(n * (1 - p_aae) * ratio(split))
        )
        n_2 = (
            lambda split: int((1 - ratio(split)) / ratio(split) * 2000)
            if split != "train"
            else int(n * (1 - p_aae) * (1 - ratio(split)))
        )
        n_3 = (
            lambda split: int((1 - ratio(split)) / ratio(split) * 2000)
            if split != "train"
            else int(n * p_aae * (1 - ratio(split)))
        )
    data_quants = [n_1, n_2, n_3, n_4]
    for split in splits_map.keys():
        X, y, protected_label = [], [], []
        for file, perc, label, protected in zip(
            files, data_quants, [1, 1, 0, 0], [1, 0, 1, 0]
        ):
            filepath = os.path.join(save_path, splits_map[split], file)
            data = np.load(filepath)
            data = list(data[: perc(split)])
            X = X + data
            y = y + [label] * len(data)
            protected_label = protected_label + [protected] * len(data)
        datasets_dict[split] = Dataset.from_dict(
            {task_to_keys["moji_preproc"][0]: X, "label": y}
        )
    datasets = DatasetDict(datasets_dict)
    return datasets


def split_to_ids(split, texts):
    if split == "train":
        return texts[:40000]
    elif split == "validation":
        return texts[40000:42000]
    elif split == "test":
        return texts[42000:44000]


def load_moji_raw(config):
    """Loads Moji dataset from fairlib (raw version for BERT/etc)."""
    from fairlib.datasets import moji
    import emoji

    save_path = os.path.join(config.cache_dir, config.data.task_name)
    dataset_class = moji.init_data_class(
        dest_folder=save_path, batch_size=config.fairlib.batch_size
    )
    splits = ["train", "dev", "test"]
    # check if we already have cached dataset
    # list of data files in folder
    files = ["pos_pos_text", "pos_neg_text", "neg_pos_text", "neg_neg_text"]
    # we have folder with files
    files_by_split = [os.path.join(save_path, file) for file in files]
    if not (all([os.path.isfile(file) for file in files_by_split])):
        dataset_class.prepare_data()
        raise NotImplementedError(f"There are no dataset files in {save_path}!")
    # now load processed data from cache_dir and transform to Dataset
    # for more info look at the preprocessing script for Moji dataset in fairlib
    datasets_dict = {}
    splits_map = {"train": "train", "validation": "dev", "test": "test"}
    # calc percentage of used data for each file
    ratio = (
        lambda split: 0.8
        if (
            (split == "train" and not config.data.get("balanced", False))
            or config.data.get("unbalance_test", False)
        )
        else 0.5
    )
    p_aae = 0.5  # proportion of the AAE
    n = 100000
    n_1 = lambda split: int(n * p_aae * ratio(split))  # happy AAE
    n_2 = lambda split: int(n * (1 - p_aae) * (1 - ratio(split)))  # happy SAE
    n_3 = lambda split: int(n * p_aae * (1 - ratio(split)))  # unhappy AAE
    n_4 = lambda split: int(n * (1 - p_aae) * ratio(split))  # unhappy SAE
    if config.data.get("unbalance_test", False):
        # recalc n_i by real test and val sizes
        # mimic train distribution
        n_1 = lambda split: 2000 if split != "train" else int(n * p_aae * ratio(split))
        n_4 = (
            lambda split: 2000
            if split != "train"
            else int(n * (1 - p_aae) * ratio(split))
        )
        n_2 = (
            lambda split: int((1 - ratio(split)) / ratio(split) * 2000)
            if split != "train"
            else int(n * (1 - p_aae) * (1 - ratio(split)))
        )
        n_3 = (
            lambda split: int((1 - ratio(split)) / ratio(split) * 2000)
            if split != "train"
            else int(n * p_aae * (1 - ratio(split)))
        )
    data_quants = [n_1, n_2, n_3, n_4]
    for split in splits_map.keys():
        X, y, protected_label = [], [], []
        for file, perc, label, protected in zip(
            files, data_quants, [1, 1, 0, 0], [1, 0, 1, 0]
        ):
            filepath = os.path.join(save_path, file)
            with open(filepath, "rb") as f:
                texts = f.readlines()
            decoded_texts = []
            for el in texts:
                try:
                    # by default we will use demojize to ensure better compatibility with BERT
                    # decoded_texts.append(emoji.demojize(el.decode()))
                    decoded_texts.append(el.decode())
                except:
                    pass
            decoded_texts = split_to_ids(split, decoded_texts)
            data = decoded_texts[: perc(split)]
            X = X + data
            y = y + [label] * len(data)
            protected_label = protected_label + [protected] * len(data)
        datasets_dict[split] = Dataset.from_dict(
            {task_to_keys["moji_raw"][0]: X, "label": y}
        )
    datasets = DatasetDict(datasets_dict)
    return datasets


def load_trustpilot(config):
    """Loads TrustPilot dataset from fairlib. Works only with already loaded data!"""
    save_path = os.path.join(config.cache_dir, config.data.task_name)
    splits = ["train", "dev", "test"]
    # check if we already have cached dataset
    # list of data files in folder
    filenames = [f"trustpilot_{split}.pkl" for split in splits]
    files = [os.path.join(save_path, filename) for filename in filenames]
    if not (all([os.path.isfile(file) for file in files])):
        raise NotImplementedError(f"There are no dataset files in {save_path}!")
    # now load processed data from cache_dir and transform to Dataset
    datasets_dict = {}
    splits = ["train", "validation", "test"]
    for file, split in zip(files, splits):
        data = pd.read_pickle(file)
        is_mlp = ("fixed" in config.model.model_name_or_path) or (
            "mlp" in config.model.model_name_or_path
        )
        if not is_mlp:
            datasets_dict[split] = Dataset.from_dict(
                {
                    task_to_keys["trustpilot"][0]: data[task_to_keys["trustpilot"][0]],
                    "label": data["target_label"].astype(np.int),
                }
            )
        else:
            datasets_dict[split] = Dataset.from_dict(
                {
                    "input_ids": data[task_to_keys["trustpilot"][1]],
                    "label": data["target_label"].astype(np.int),
                }
            )
    datasets = DatasetDict(datasets_dict)
    return datasets


def get_protected_attribute(filepath, label_name="gender_class"):
    data = pd.read_pickle(filepath)
    return data[label_name].values


def load_jigsaw_race(config):
    """Loads Jigsaw dataset from fairlib."""
    from fairlib.datasets import jigsaw

    save_path = os.path.join(config.cache_dir, config.data.task_name)
    if config.data.get("task_dir", False):
        save_path = os.path.join(config.cache_dir, config.data.task_dir)
    log.info(save_path)
    batch_size = 16 if not config.get("fairlib", False) else config.fairlib.batch_size
    dataset_class = jigsaw.init_data_class(dest_folder=save_path, batch_size=batch_size)
    split_names = {"train": "train", "dev": "val", "test": "test"}
    splits = ["train", "dev", "test"]
    # check if we already have cached dataset
    # list of data files in folder
    files = [
        os.path.join(save_path, "jigsaw_race_{}.pq".format(split_names[split]))
        for split in split_names.keys()
    ]
    if not (all([os.path.isfile(file) for file in files])):
        dataset_class.prepare_data()
    # now load processed data from cache_dir and transform to Dataset
    datasets_dict = {}
    splits = ["train", "validation", "test"]
    for file, split in zip(files, splits):
        data = pd.read_parquet(file)
        is_mlp = ("fixed" in config.model.model_name_or_path) or (
            "mlp" in config.model.model_name_or_path
        )
        if not is_mlp:
            datasets_dict[split] = Dataset.from_dict(
                {
                    task_to_keys["jigsaw_race"][0]: data[
                        task_to_keys["jigsaw_race"][0]
                    ],
                    "label": data["binary_label"].astype(np.int),
                }
            )
        else:
            datasets_dict[split] = Dataset.from_dict(
                {
                    "input_ids": data[task_to_keys["jigsaw_race"][1]],
                    "label": data["binary_label"].astype(np.int),
                }
            )
    datasets = DatasetDict(datasets_dict)
    return datasets
