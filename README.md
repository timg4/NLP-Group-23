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

### Running Milestone 2 Baselines
```bash
python milestone2/milestone2.py --data data/manual_annotation/sample_sentences_labeled.conllu
```

This will:
- Train and evaluate 3 baseline NER methods (80/20 train/dev split)
- Generate predictions in CoNLL-U format
- Save metrics, error analysis, and example sentences to `milestone2/results/`

---

## Milestone 1: Data Preprocessing

### Dataset

**FinCorpus-DE10k**: A German financial/legal corpus containing 10,000 PDF documents with extracted text (165M+ tokens). Documents include annual reports, base prospectuses, final terms, and legal texts.

**Source**: `anhaltai/fincorpus-de-10k` (HuggingFace)

### Preprocessing Pipeline

Initial experimentation was conducted in `data/preprocess.ipynb` on a small subset.

The preprocessing script (`data/perprocess.py`) performs:

1. **Text normalization**:
   - Whitespace cleanup
   - Paragraph sign (§) normalization
   - Abbreviation protection (e.g., "z. B.", "bzw.", "Art.") to prevent incorrect sentence segmentation

2. **NLP processing** (Stanza German pipeline):
   - Sentence segmentation
   - Tokenization
   - POS tagging
   - Lemmatization

3. **CoNLL-U export**: All documents merged into a single file

### Usage

```bash
python data/perprocess.py --output data/preprocessing/processed/fincorpus_processed.conllu --max_docs 1000
```

**Current corpus**: 1,000 documents (randomly sampled from all collections)
**Processing time**: ~5.5 hours

### Output Format

Data stored in CoNLL-U format with standard 10-column structure. Includes token IDs, word forms, lemmas, POS tags (UPOS + German STTS), morphological features, and character offsets.

Documents delimited with `# newdoc id = <doc_id>` headers.

**Download**: File too large for GitHub. Available at: [Google Drive](https://drive.google.com/file/d/1bs7oI4dxBr2b7Hdp_BC9zWBHjSzHUCVl/view?usp=share_link)
A glimpse into the first 2000 lines can be found at `data/preprocessing/processed/first2000.txt`.

### Issues Identified

1. **Processing performance**: Current throughput is ~10 docs/minute. Full corpus would take ~17 hours. Limited to 1000 docs for this project.

2. **HTML artifacts**: HTML entities like `&quot;` appear in output instead of standard quotes.

3. **PDF formatting**: Multiple consecutive newlines (e.g., `SpacesAfter=\s\n\s\n\s\n`) due to formatting from original PDFs.

4. **Dependency parsing**: Not included in current version (columns 7-9 are placeholders). Adding `depparse` processor would approximately double processing time.

Issues 1-3 are cosmetic and could be addressed in future steps if necessary. Issue 4 was skipped due to time constraints. Overall, the file is clean and ready for downstream tasks.

---

## Milestone 2: Baseline NER Methods

We implemented three baseline methods for German financial NER with entity types ORGANIZATION (ORG), MONETARY (MON), and LEGAL_REFERENCE (LEG). The dataset consists of 150 manually annotated sentences split 80/20 into train and dev sets.

The three methods are SimpleRuleNER (pattern-based rules), TokenNB with data priors (Naive Bayes using empirical label frequencies), and TokenNB with uniform priors (Naive Bayes with equal prior probabilities for all labels).

### Results

SimpleRuleNER achieved the best overall performance with 0.920 accuracy and 0.926 weighted F1. It performed well on monetary values (F1: 0.654) and legal references (F1: 0.667 for B-LEG tags) but completely failed on organizations (F1: 0.000). TokenNB with data priors predicted only 'O' labels due to severe class imbalance in the training data (~93% 'O' tags), resulting in 0% recall on all entity types despite high overall accuracy (0.928). TokenNB with uniform priors performed better by treating all labels equally, achieving the best organization detection (F1: 0.667) but at the cost of lower overall accuracy (0.784) due to many false positives.

Performance comparison on the 30-sentence dev set:

| Method | Accuracy | Macro F1 | ORG F1 | MON F1 | LEG F1 |
|--------|----------|----------|--------|--------|--------|
| SimpleRuleNER | 0.920 | 0.489 | 0.000 | 0.654 | 0.667 |
| TokenNB (data priors) | 0.928 | 0.138 | 0.000 | 0.000 | 0.000 |
| TokenNB (uniform priors) | 0.784 | 0.400 | 0.667 | 0.100 | 0.571 |

### Example Predictions

Example showing organization detection failure:

Sentence: "Berechnungsstelle : UniCredit Bank AG , Apianstr ."

```
Token          Gold      SimpleRuleNER    TokenNB (data)    TokenNB (uniform)
UniCredit      B-ORG     O                O                 O
Bank           I-ORG     O                O                 O
AG             I-ORG     O                O                 O
```

All three methods failed to detect this organization. SimpleRuleNER only has patterns for specific keywords like "sparkasse" and suffixes like "AG" in isolation. The Naive Bayes models either predict only 'O' (data priors) or make random guesses (uniform priors) without understanding multi-word entities.

Example showing legal reference detection:

Sentence: "Moody's is established in the European Community and registered since 31 October 2011 under the CRA Regulation ."

```
Token          Gold      SimpleRuleNER
CRA            B-LEG     O
Regulation     I-LEG     O
```

SimpleRuleNER missed this legal reference because its patterns only look for "§", "Artikel", and "Art." followed by numbers. Legal abbreviations like "CRA Regulation" are not covered.

Example showing false positive on monetary:

Sentence: "112 Rechte der Anteilsinhaber ..."

```
Token    Gold    SimpleRuleNER
112      O       B-MON
```

SimpleRuleNER incorrectly tagged "112" as monetary. The pattern matches any number with separators or percentages, but cannot distinguish section numbers from actual amounts.

### Analysis

The main issue with SimpleRuleNER is insufficient pattern coverage. It works well for stereotypical cases like "§ 15" or "3,41 %" but misses variations. The keyword list for organizations is too limited and doesn't handle multi-word company names properly.

TokenNB with data priors demonstrates a critical problem with class imbalance. When 93% of training tokens are 'O', the prior probability P(O) ≈ 0.93 overwhelms any token-label associations. Even strong indicators like "§" for legal references get outweighed by the prior, causing the model to predict only 'O'.

TokenNB with uniform priors (P(label) = 1/7) forces the model to rely on token likelihoods rather than priors. This helps detect entities but introduces many errors because single-token features cannot capture multi-word entities or sequential dependencies. Accuracy drops from 0.928 to 0.784, but macro F1 improves from 0.138 to 0.400.

Token-level Naive Bayes is fundamentally inadequate for NER because it treats each token independently. BIO tagging requires understanding sequences (B- must precede I- tags), and multi-word entities require context beyond single tokens. The class imbalance problem could be partially addressed with techniques like oversampling rare classes or using class weights, but the independence assumption remains a core limitation.

For future work, sequence models like CRF or BiLSTM-CRF would better handle the sequential dependencies and context needed for accurate NER. Rule-based methods could be improved by expanding pattern coverage and adding context checks to reduce false positives.

Detailed outputs are available in `milestone2/results/` including full predictions, error breakdowns by entity type, and classification reports.

---

## Final Submission

After milestone 2 we reworked the manually labeled data to improve quality and extended it to 170 sentences. To make this process more efficient, we developed a GUI annotation tool (`data/manual_annotation/anotate_conllu_windows.py`) that allows token-by-token labeling with keyboard shortcuts, BIO validation, and autosave functionality.

### Labeling Pipeline

A major part of our final submission was investigating a pipeline for going from unlabeled data to a reliable labeled dataset. The pipeline we propose consists of:

1. **Hand labeling** - Create a small gold standard dataset manually
2. **LLM generation** - Use an LLM API to generate labels for additional data
3. **Validation** - Check LLM-generated labels against the gold standard
4. **Training** - Use the validated labeled data to train ML models

### Main Approach: RuleChef

Our main focus was on RuleChef, an approach that uses LLMs to learn explicit extraction rules iteratively. The idea is quite nice: train on gold examples, let the LLM synthesize rules, evaluate on false positives/negatives, and refine the rules until you have an interpretable ruleset.

However, even after spending a lot of time tuning the task descriptions and feedback loops, we could not get RuleChef to perform well on our data. The main problems were:
- Refinement often overgeneralized
- Formatting issues with malformed rules or invalid JSON
- Struggles with unstructured classes like ORG
- Inconsistent results across different runs

### Comparison Approaches

Since RuleChef did not work as expected, we briefly tried some other approaches without too much tuning:

| Method | Description |
|--------|-------------|
| **EnhancedRuleNER** | Hand-crafted regex patterns with POS-tag validation heuristics |
| **TokenNB** | Token-level Naive Bayes with uniform priors to handle class imbalance |
| **CRF** | Linear-chain CRF with rich token features (casing, prefixes, domain-specific features) |
| **SpaCy + Heuristics** | Hybrid approach combining spaCy German NER with keyword-based rules |
| **German BERT** | Fine-tuned bert-base-german-cased for token classification (trained on only 150 sentences) |
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
python final_submission/run_rulechef.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_milestone2_nb_uniform_priors.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_enhanced_rule_based.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_crf.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_project_labeling.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_bert_ner.py --data data/manual_annotation/hand_labelled.conllu
python final_submission/run_openai_ner.py --data data/manual_annotation/hand_labelled.conllu  # requires API key
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
│   ├── README.md                       # Final submission documentation
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
├── future_work/                        # Advanced methods (beyond M2)
│   ├── README_FUTURE_WORK.md          # Documentation
│   ├── baselines/                      # CRF methods
│   ├── evaluation/                     # Comparison scripts
│   ├── results/                        # ChatGPT vs Project comparison
│   └── data/                           # spaCy + regex hybrid
│
└── NLP_IE_2025WS_Exercise-2.pdf       # Assignment description
```

---

## Notes

- Milestone 1 & 2 represent the core project requirements
- Final submission extends the project with RuleChef experiments and multiple comparison approaches
- Final evaluation uses 170 manually-labeled sentences with stratified 80/20 train/test split (seed 2323)
- See `final_submission/results/summary.txt` for the combined results table
- Due to labeling ambiguity (e.g. whether currency symbols belong to monetary values), overlap metrics are often more informative than exact match F1

---

## AI Usage Disclaimer

We used AI assistance in the following parts of this project:
- **Code completion**: VSCode inline suggestions (GitHub Copilot) for faster coding
- **GUI labeling tool**: The annotation tool (`anotate_conllu_windows.py`) was partly developed with AI assistance
- **Documentation**: AI helped write and structure parts of this README and code comments
