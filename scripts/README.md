# How to run experiments
Algorithm:
1. You are able to find the optimal model hyperparameters by yourself using script `electra_hp_search_hue_datasets.sh`
2. Run model training on 80\% of the datasets for tuning HUQ hyperparameters using script `run_train_electra_models_toxic_method_hp.sh` and evaluation with script `run_eval_electra_models_toxic_method_hp.sh`
3. Run model training on the entire datasets using script `run_train_models_electra_toxic.sh` and evaluation with script `run_eval_electra_toxic_huq.sh`
4. Run model ensembles with `run_train_ensemble_models_toxic.sh` and `run_eval_ensemble_series_toxic.sh`

You can also regenerate all these scripts with other hyperparameters using [notebook](/src/exps_notebooks/generate_series_of_exps_huq_final.ipynb)