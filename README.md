# NLP Project - German Financial NER

**Topic 10: Named Entity Recognition for German Official Documents**

Comparing baseline NER methods for German financial/legal documents with three entity types: ORGANIZATION (ORG), MONETARY (MON), and LEGAL_REFERENCE (LEG).

---

## Setup

### Prerequisites
- Python 3.11 or higher
- Virtual environment recommended

### Installation
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset & Preprocessing (Milestone 1)

**FinCorpus-DE10k**: German financial/legal corpus with 10,000 PDF documents from HuggingFace ([anhaltai/fincorpus-de-10k](https://huggingface.co/datasets/anhaltai/fincorpus-de-10k)). We sampled 1,000 documents and processed them with Stanza (tokenization, POS tagging, lemmatization) into CoNLL-U format.

**Processed corpus download**: [Google Drive](https://drive.google.com/file/d/1bs7oI4dxBr2b7Hdp_BC9zWBHjSzHUCVl/view?usp=share_link)

For detailed milestone documentation, see [archive/README_MILESTONES.md](archive/README_MILESTONES.md).

---

## Baseline Methods (Milestone 2)

For milestone 2 we implemented three baseline methods: SimpleRuleNER, TokenNB with data priors, and TokenNB with uniform priors. The baselines showed that rule-based methods work okay for structured entities (MON, LEG) but fail on variable entities like ORG. Naive Bayes suffers from class imbalance (~93% O-tags).

For detailed results, examples and analysis, see [archive/README_MILESTONES.md](archive/README_MILESTONES.md).

---

## Final Submission

After milestone 2 we reworked the manually labeled data to improve quality and extended it to 170 sentences. To make this process more efficient, we developed a GUI annotation tool (`data/manual_annotation/anotate_conllu_windows.py`) that allows token-by-token labeling with keyboard shortcuts, BIO validation, and autosave functionality.

### Labeling Pipeline

In addition to the main NER task, we investigated a pipeline for going from unlabeled data to a reliable labeled dataset. The pipeline we propose consists of:

1. **Hand labeling** - Create a small gold standard dataset manually
2. **LLM generation** - Use an LLM API to generate labels for additional data
3. **Validation** - Check LLM-generated labels against the gold standard
4. **Training** - Use the validated labeled data to train ML models

### Main Approach: RuleChef

Our main focus was on RuleChef ([KRLabsOrg/rulechef](https://github.com/KRLabsOrg/rulechef)), an approach that uses LLMs to learn explicit extraction rules iteratively. The idea is quite nice: train on gold examples, let the LLM synthesize rules, evaluate on false positives/negatives, and refine the rules until you have an interpretable ruleset.

However, even after spending a lot of time tuning the task descriptions and feedback loops, we could not get RuleChef to perform well on our data. The main problems were:
- Refinement often overgeneralized
- Formatting issues with malformed rules or invalid JSON
- Struggles with unstructured classes like ORG
- Inconsistent results across different runs

### RuleChef Rule Examples

Rules are stored under `models/rule-chef-v2/rulechef_v2_data/`. One full code rule and one short regex rule:

**Rule 1 (ORG, code - full rule)**
Name: "Exclude generic capitalized nouns"
```python
def extract(input_data):
    import re
    spans = []
    text = input_data['text']
    regex_pattern = r'\b([A-ZÄÖÜ][a-zäöüß]+(?:\s[A-ZÄÖÜ][a-zäöüß]*)*?(?:\s+(AG|GmbH|KG|SE|Bank|Sparkasse|e\.V\.|Aktiengesellschaft|GesellschaftmbH|Group|Holding|Service|Agency|Corporate|Association))\b)'
    matches = re.finditer(regex_pattern, text)
    for match in matches:
        spans.append({
            'start': match.start(),
            'end': match.end(),
            'text': match.group(0)
        })
    return spans
```

**Rule 2 (MON, regex)**
Name: "Capture all Monetary References"
Pattern:
```regex
\d+(?:[.,]\d+)?\s?(?:EUR|USD|Mio|Mrd|\$|GBP)
```

### Invalid Rule Example (skipped)

During MON rule synthesis, RuleChef produced a code rule with a stray `\n` after the regex literal, which caused a Python syntax error and the rule was skipped:
```python
pattern = r'\b(?:EUR|USD|GBP)\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?'\n
matches = re.finditer(pattern, text)
```

### RuleChef Qualitative Examples

The examples below are from the dev split used in the final evaluation (seed 2323).

**Example 1 (MON boundary error + false positives)**
Sentence: "Die Stärkung der Kapitalquoten soll durch eine Barkapitalzufuhr in Höhe von EUR 2,835 Mrd., an der sich die Bundesländer Niedersach- sen und Sachsen-Anhalt zusammen mit insgesamt EUR 1,7 Mrd. be- teiligen sollen, durchgeführt werden."
Gold: MON = "EUR 2,835 Mrd.", "EUR 1,7 Mrd."
RuleChef: MON = "2,835 Mrd.", "EUR 1,7 Mrd."; ORG = "Die Stärkung", "Kapitalquoten", "Barkapitalzufuhr"; LEG = "in Höhe von EUR", "Bundesländer Niedersach-"

**Example 2 (ORG noise)**
Sentence: "Das bauspartechnische Risiko ist eng mit dem Geschäftsmodell der BSH verknüpft und kann daher nicht vermieden werden."
Gold: ORG = "BSH"
RuleChef: ORG = "Das", "Geschäftsmodell", "BSH"; MON = "bauspartechnische Risiko"

**Example 3 (LEG correct, many false positives)**
Sentence: "3. Wenn der Fälligkeitstag oder ein Zinszahltag kein Bankgeschäftstag gemäß § 2 ist, so besteht der Anspruch der Schuldverschreibungsgläubiger auf Zahlung erst an dem nächstfolgenden Bankgeschäftstag."
Gold: LEG = "§ 2"
RuleChef: LEG = "§ 2"; ORG = "Wenn", "Fälligkeitstag", "Zinszahltag", "Bankgeschäftstag", "Anspruch"; MON = "Schuldverschreibungsgläubiger auf Zahlung erst an dem"

### Comparison Approaches

Since RuleChef did not work as expected, we briefly tried some other approaches without too much tuning:

| Method | Description |
|--------|-------------|
| **EnhancedRuleNER** | Hand-crafted regex patterns with POS-tag validation heuristics |
| **TokenNB** | Token-level Naive Bayes with uniform priors to handle class imbalance |
| **CRF** | Linear-chain CRF with rich token features (casing, prefixes, domain-specific features) |
| **SpaCy + Heuristics** | Hybrid approach combining spaCy German NER with keyword-based rules |
| **German BERT** | Fine-tuned [bert-base-german-cased](https://huggingface.co/google-bert/bert-base-german-cased) for token classification (trained on only 150 sentences) |
| **OpenAI API** | Direct prompting of GPT models for NER (expensive but good results) |

### Results

Final results on the test set (30 sentences, stratified 80/20 split):

| Method | LEG F1 | MON F1 | ORG F1 | Macro F1 | Accuracy | Time (s) |
|--------|--------|--------|--------|----------|----------|----------|
| OpenAI API | 1.000 | 1.000 | 0.588 | 0.894 | 0.975 | 167.95 |
| SpaCy + Heuristics | 0.857 | 0.824 | 0.440 | 0.774 | 0.956 | 0.24 |
| BERT | 0.429 | 0.692 | 0.615 | 0.679 | 0.963 | 99.52 |
| CRF | 0.667 | 0.970 | 0.056 | 0.667 | 0.951 | 0.76 |
| EnhancedRuleNER | 0.588 | 0.667 | 0.200 | 0.605 | 0.933 | 0.01 |
| TokenNB | 0.067 | 0.897 | 0.200 | 0.513 | 0.803 | 0.01 |
| RuleChef | 0.071 | 0.098 | 0.012 | 0.222 | 0.541 | 370.75 |

**Note:** The F1 scores above are exact match metrics. Due to ambiguity in what exactly gets labeled (e.g., should "EUR" be part of the monetary value or not), overlap metrics usually give a better picture of actual performance. See `final_submission/results/` for detailed overlap metrics per method.

### Insights

- NER for German official documents is a trade-off problem - there is no free lunch
- Entity structure matters a lot: ORG consistently underperforms across all methods because organization names are very variable
- RuleChef was unstable and ineffective in our setup, though it might work better with different data or more tuning
- The OpenAI API gives the best results but is expensive and slow
- SpaCy combined with domain-specific heuristics offers a good balance between performance and cost

### Running the Final Submission

From the repo root, run each model separately:

```bash
python final_submission/run_rulechef.py --data data/manual_annotation/hand_labelled.conllu   # requires OPENAI API key
python final_submission/run_milestone2_nb_uniform_priors.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_enhanced_rule_based.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_crf.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_project_labeling.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_bert_ner.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_openai_ner.py --data data/manual_annotation/hand_labelled.conllu  # requires OpenAI API key
```

To generate a combined summary:

```bash
python final_submission/utilities/summarize_results.py
```

Outputs are saved to `final_submission/results/<ModelName>/`.

---

## Project Structure

```
NLP-Group-23/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore
│
├── data/
│   ├── preprocessing/fincorpus-de-10k.py   # Dataset loader
│   ├── perprocess.py                   # M1: Preprocessing script
│   ├── preprocess.ipynb                # M1: Preprocessing experimentation
│   ├── sample_for_manual_annotation.py # Sampling for annotation
│   ├── processed/                      # M1 output (CoNLL-U corpus)
│   │   ├── fincorpus_processed.conllu  # Too large for GitHub
│   │   ├── fincorpus_processed.conllu.zip
│   │   └── first2000.txt               # Preview
│   └── manual_annotation/              # Gold standard annotations
│       ├── hand_labelled.conllu        # Final labeled dataset (170 sentences)
│       ├── sample_sentences.conllu
│       └── anotate_conllu_windows.py   # GUI labeling tool
│
├── milestone2/
│   ├── milestone2.py                   # M2: Main baseline script
│   ├── experimentation.ipynb           # M2: Exploration notebook
│   └── results/                        # M2 outputs
│       ├── rule_based_predictions.conllu
│       ├── nb_data_priors_predictions.conllu
│       ├── nb_uniform_priors_predictions.conllu
│       ├── error_analysis.txt
│       ├── metrics_summary.txt
│       └── example_sentences.txt
│
├── final_submission/                   # Final submission comparison
│   ├── run_rulechef.py                 # RuleChef runner
│   ├── run_bert_ner.py                 # German BERT runner
│   ├── run_crf.py                      # CRF runner
│   ├── run_openai_ner.py               # OpenAI API runner
│   ├── run_project_labeling.py         # SpaCy + heuristics runner
│   ├── run_enhanced_rule_based.py      # Enhanced rule-based runner
│   ├── run_milestone2_nb_uniform_priors.py  # TokenNB runner
│   ├── utilities/
│   │   ├── common.py                   # Shared utilities
│   │   ├── stratified_split.py         # Train/test split
│   │   └── summarize_results.py        # Generate summary table
│   └── results/                        # Per-model outputs
│       ├── summary.txt                 # Combined results table
│       ├── BERT/
│       ├── CRF/
│       ├── RuleChef/
│       ├── OpenAI_NER/
│       ├── SpacyLabeling/
│       ├── EnhancedRuleBasedNER/
│       └── TokenNB_uniform_priors/
│
├── archive/                            # Milestone documentation
│   ├── README_MILESTONES.md            # Detailed M1 & M2 documentation
│   └── future_work/                    # Early experiments (beyond M2)
│
└── NLP_Group23_Presentation.pdf       # Final Presentation
```

---

## Notes

- Final evaluation uses 170 manually-labeled sentences with stratified 80/20 train/test split (seed 2323)
- See `final_submission/results/summary.txt` for the combined results table
- Due to labeling ambiguity (e.g. whether currency symbols belong to monetary values), overlap metrics are often more informative than exact match F1

---

## AI Usage Disclaimer

We used AI assistance in the following parts of this project:
- **Code completion**: VSCode inline suggestions (GitHub Copilot) for faster coding
- **GUI labeling tool**: The annotation tool (`anotate_conllu_windows.py`) was partly developed with AI assistance
- **Documentation**: AI helped write and structure parts of this README and code comments
