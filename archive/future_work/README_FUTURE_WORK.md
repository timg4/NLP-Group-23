# Future Work - Advanced NER Methods

This folder contains NER methods and experiments that go beyond the Milestone 2 baseline requirements.

## Contents

### baselines/
Advanced NER implementations:
- `crf_ner.py` - Conditional Random Field (CRF) based NER
- `rule_based_ner.py` - Advanced rule-based NER
- `train_crf.py` - CRF model training script
- `train_rule_based.py` - Rule-based method training

### evaluation/
- `compare_labeling_methods.py` - Script for comparing multiple labeling approaches

### results/
Comparison results from evaluating different labeling methods:
- `gpt_labeling_approach.py` - GPT-based labeling implementation
- `labeling_comparison/` - Detailed comparison outputs:
  - ChatGPT vs Project method (spaCy + regex hybrid)
  - Performance metrics, visualizations, and analysis

### data/
- `project_labeling.py` - Hybrid NER method combining spaCy ML model with regex patterns and keyword matching

## Key Results from Advanced Methods

### ChatGPT vs Project Method Comparison

**Performance Summary** (F1 Scores):

| Method | Overall F1 | ORG | MON | LEG |
|--------|-----------|-----|-----|-----|
| ChatGPT | 0.3985 | 0.4186 | 0.2381 | 0.5333 |
| **Project (Improved)** | **0.4108** | 0.3967 | **0.3059** | **0.5333** |

**Key Findings:**
- Project method (spaCy + regex + keywords) outperforms ChatGPT by 3.1% overall F1
- Best at monetary detection (+28.5% better than ChatGPT)
- Ties on legal references (F1: 0.53)
- 54.9% improvement over initial baseline (0.2652 → 0.4108)
- Better precision-recall balance (Recall: 0.36 vs ChatGPT: 0.31)

**Methodology:** The Project method uses a hybrid approach:
- spaCy German NER model for organization detection
- Regex patterns for legal references and monetary values
- Keyword-based organization matching
- Token-level pattern matching

## Running Advanced Methods

### CRF-based NER
```bash
python future_work/baselines/train_crf.py --input <data_file>
python future_work/baselines/crf_ner.py --model <trained_model> --input <test_file>
```

### Project Method (spaCy + Regex)
```bash
python future_work/data/project_labeling.py \
    --input data/manual_annotation/sample_sentences.conllu \
    --output <output_file>
```

### Compare Methods
```bash
python future_work/evaluation/compare_labeling_methods.py \
    --gold data/manual_annotation/sample_sentences_labeled.conllu \
    --chatgpt future_work/results/labeling_comparison/chatgpt_predictions.tsv \
    --project future_work/results/labeling_comparison/project_predictions.conllu \
    --output <output_directory>
```

## Notes

- These methods represent explorations beyond the Milestone 2 baseline requirements
- Milestone 2 focuses on SimpleRuleNER and TokenNB (Naive Bayes) baselines only
- See main README.md for Milestone 2 baseline results and documentation
