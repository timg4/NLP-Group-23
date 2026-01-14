# Final Submission - Unified Comparison

This folder contains per-model evaluation scripts that all use the same
stratified 80/20 split (seed 2323) and write comparable metrics and predictions.
Metrics are token-level and collapsed to entity labels (LEG, MON, ORG, O).
Each run also reports span-level overlap metrics (per label + micro).

## What it compares
- RuleChef (`models/rule-chef-v2`)
- Milestone 2 baselines (copied to `models/milestone2`)
- Enhanced rule-based NER (`models/enhanced_rulebased_NER`)
- OpenAI NER (`models/openai-ner`, optional, requires API key)

## Usage
From the repo root (run each model separately):

```
python final_submission/run_rulechef.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_milestone2_simple_rule.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_milestone2_nb_uniform_priors.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_future_rule_based.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_openai_ner.py --data data/manual_annotation/hand_labelled.conllu
```

## Outputs
- `final_submission/results/<ModelName>/metrics_summary.txt`
- `final_submission/results/<ModelName>/metrics.json`
- `final_submission/results/<ModelName>/predictions.conllu`

To build a combined summary after running individual models:

```
python final_submission/utilities/summarize_results.py
```


