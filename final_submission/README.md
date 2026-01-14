# Final Submission - Unified Comparison

This folder contains per-model evaluation scripts that all use the same
stratified 80/20 split (seed 2323) and write comparable metrics and predictions.
Metrics are token-level and collapsed to entity labels (LEG, MON, ORG, O).

## What it compares
- RuleChef (rule-chef-v2)
- Milestone 2 baselines (SimpleRuleNER, TokenNB data priors, TokenNB uniform priors)
- Future work rule-based NER baseline
- OpenAI NER (optional, requires API key)

## Usage
From the repo root (run each model separately):

```
python final_submission/run_rulechef.py --data data/manual_annotation2/my_labels.conllu
python final_submission/run_milestone2_simple_rule.py --data data/manual_annotation2/my_labels.conllu
python final_submission/run_milestone2_nb_data_priors.py --data data/manual_annotation2/my_labels.conllu
python final_submission/run_milestone2_nb_uniform_priors.py --data data/manual_annotation2/my_labels.conllu
python final_submission/run_future_rule_based.py --data data/manual_annotation2/my_labels.conllu
python final_submission/run_openai_ner.py --data data/manual_annotation2/my_labels.conllu
```

## Outputs
- `final_submission/results/<ModelName>/metrics_summary.txt`
- `final_submission/results/<ModelName>/metrics.json`
- `final_submission/results/<ModelName>/predictions.conllu`

To build a combined summary after running individual models:

```
python final_submission/summarize_results.py
```
