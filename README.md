# Hybrid Uncertainty Quantification for Selective Text Classification in Ambiguous Tasks

This repository contains the implementation of hybrid methods for Uncertainty Estimation (UE) for selective text classification tasks on several ambiguous datasets (toxicity detection, sentiment analysis, and multi-class classification) based on the Transformer models for NLP. 

Namely, the repository contains code for the paper ["Hybrid Uncertainty Quantification for Selective Text Classification in Ambiguous Tasks"](https://aclanthology.org/2023.acl-long.652/) at the ACL-2023 conference.

# What the paper is about?
Potential mistakes in automated classification can be identified by using uncertainty estimation (UE) techniques. Although UE is a rapidly growing field within natural language processing, we find that state-of-the-art UE methods estimate only epistemic uncertainty and show poor performance, or under-perform trivial methods for ambiguous tasks such as toxicity detection. We argue that in order to create robust uncertainty estimation methods for ambiguous tasks it is necessary to account also for aleatoric uncertainty. 

In this paper, we propose a new uncertainty estimation method that combines epistemic and aleatoric UE methods. We show that by using our hybrid method, we can outperform state-of-the-art UE methods for toxicity detection and other ambiguous text classification tasks.


# Examples
The toy example with HUQ on the two moons dataset could be found in the [jupyter notebook](src/exps_notebooks/two_moons.ipynb). 

Example scripts for training and evaluating models with calculating their UE scores could be found in the [scripts/miscl_scripts](scripts/miscl_scripts) directory. This directory also contains instructions for reproducing results from the paper.

Configuration files with parameters of models, datasets, and uncertainty estimation methods are located in [configs](configs).

# Citation
```bibtex
@inproceedings{vazhentsev-etal-2023-hybrid,
    title = "Hybrid Uncertainty Quantification for Selective Text Classification in Ambiguous Tasks",
    author = "Vazhentsev, Artem  and
      Kuzmin, Gleb  and
      Tsvigun, Akim  and
      Panchenko, Alexander  and
      Panov, Maxim  and
      Burtsev, Mikhail  and
      Shelmanov, Artem",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.652",
    pages = "11659--11681",
    abstract = "Many text classification tasks are inherently ambiguous, which results in automatic systems having a high risk of making mistakes, in spite of using advanced machine learning models. For example, toxicity detection in user-generated content is a subjective task, and notions of toxicity can be annotated according to a variety of definitions that can be in conflict with one another. Instead of relying solely on automatic solutions, moderation of the most difficult and ambiguous cases can be delegated to human workers. Potential mistakes in automated classification can be identified by using uncertainty estimation (UE) techniques. Although UE is a rapidly growing field within natural language processing, we find that state-of-the-art UE methods estimate only epistemic uncertainty and show poor performance, or under-perform trivial methods for ambiguous tasks such as toxicity detection. We argue that in order to create robust uncertainty estimation methods for ambiguous tasks it is necessary to account also for aleatoric uncertainty. In this paper, we propose a new uncertainty estimation method that combines epistemic and aleatoric UE methods. We show that by using our hybrid method, we can outperform state-of-the-art UE methods for toxicity detection and other ambiguous text classification tasks.",
}
```

# License
© 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research Institute" (AIRI). All rights reserved.

Licensed under the [MIT License](LICENSE)