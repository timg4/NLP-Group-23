# NLP Project - German Financial NER

**Approach:** Compare 3 NER labeling methods on 150 sentences to determine which is most accurate for German financial/legal documents.

---

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
python -m spacy download de_core_news_lg
```

### Workflow

**Step 1: Sample 150 sentences**
```bash
python data/sample_for_manual_annotation.py \
    --input data/processed/fincorpus_processed.conllu \
    --output data/manual_annotation \
    --num_sentences 150
```

**Step 2: Manual labeling**
- Open `data/manual_annotation/sample_sentences_labeled.conllu`
- Review/correct NER tags in 11th column (B-ORG, I-ORG, B-MON, I-MON, B-LEG, I-LEG, O)
- Read `data/manual_annotation/labeling_guidelines.md` for instructions

**Step 3: ChatGPT labeling**
- Upload sentences to ChatGPT interface
- Get NER predictions using rule-based approach (see `results/gpt_labeling_approach.py`)
- Note: Due to file size constraints, a rule-based function was used with GPT to generate predictions
- Output: `results/labeling_comparison/chatgpt_predictions.tsv` (TSV format with columns: sent_id, token_id, token, ner_tag)

**Step 4: Project labeling**
```bash
python data/project_labeling.py \
    --input data/manual_annotation/sample_sentences.conllu \
    --output results/labeling_comparison/project_predictions.conllu
```

**Step 5: Compare all methods**
```bash
python evaluation/compare_labeling_methods.py \
    --gold data/manual_annotation/sample_sentences_labeled.conllu \
    --chatgpt results/labeling_comparison/chatgpt_predictions.tsv \
    --project results/labeling_comparison/project_predictions.conllu \
    --output results/labeling_comparison
```

**Note**: The `--chatgpt` parameter accepts both TSV and CoNLL-U formats (auto-detected by file extension).

---

## Entity Types

1. **ORGANIZATION (ORG)**: Companies, banks (e.g., "Deutsche Bank AG", "LBBW")
2. **MONETARY (MON)**: Currency amounts, percentages (e.g., "EUR 1000", "3,41 %")
3. **LEGAL_REFERENCE (LEG)**: Legal citations (e.g., "§ 15", "Art. 12 PVO")

---

## Results

After comparison, you'll get:
- `labeling_comparison.csv` - Comparison table
- `labeling_comparison.md` - Markdown table
- `f1_comparison.png` - F1 scores chart
- `detailed_report.txt` - Full analysis

---

# Milestone 2: Baseline NER Methods & Evaluation

## Baseline Approaches

Three NER methods were implemented and compared on 150 manually-annotated sentences:

1. **Manual Labeling** (Gold Standard): Human-annotated sentences following detailed guidelines
2. **ChatGPT Approach**: Rule-based method using pattern matching (see `results/gpt_labeling_approach.py`)
3. **Project Method** (ML + Rules): Hybrid approach combining:
   - spaCy German NER model (ML)
   - Regex patterns for LEG/MON entities
   - Keyword-based ORG detection
   - Token-level pattern matching

## Evaluation Results

**Performance Summary** (F1 Scores):

| Method | Overall F1 | ORG | MON | LEG |
|--------|-----------|-----|-----|-----|
| ChatGPT | 0.3985 | 0.4186 | 0.2381 | 0.5333 |
| **Project (Improved)** | **0.4108** | 0.3967 | **0.3059** | **0.5333** |

**Key Findings:**
- ✅ Project method **outperforms ChatGPT** by 3.1% overall F1
- ✅ **Best at monetary detection** (+28.5% better than ChatGPT)
- ✅ **Ties on legal references** (F1: 0.53)
- ✅ **54.9% improvement** over initial baseline (0.2652 → 0.4108)
- ✅ Better precision-recall balance (Recall: 0.36 vs ChatGPT: 0.31)

**Methodology:** Token-level pattern matching for legal references, keyword-based organization detection, and enhanced monetary detection with magnitude words significantly improved performance over basic regex patterns.

See `results/labeling_comparison/` for detailed metrics and visualizations.

---

# Milestone 1: Data Preprocessing

## Dataset

**FinCorpus-DE10k**: A German financial/legal corpus containing 10,000 PDF documents with extracted text (165M+ tokens). Documents include annual reports, base prospectuses, final terms, and legal texts.

Source: `anhaltai/fincorpus-de-10k` (HuggingFace)

## Preprocessing Pipeline

In (`data/perprocess.ipynb`) some basic experimantation was done on a small subset.

The real preprocessing script (`data/perprocess.py`) performs:

1. **Text normalization**:
   - Whitespace cleanup
   - Paragraph sign (§) normalization as we have legal data here
   - Abbreviation protection (e.g., "z. B.", "bzw.", "Art.") to prevent incorrect sentence segmentation

2. **NLP processing** (Stanza German pipeline):
   - Sentence segmentation
   - Tokenization
   - POS tagging
   - Lemmatization

3. **CoNLL-U export**: All documents merged into a single file

### Usage

```bash
python data/perprocess.py --output data/processed/fincorpus_processed.conllu --max_docs 1000
```

**Current corpus**: 1,000 documents (randomly sampled from all collections)
**Processing time**: ~5.5 hours

## Output Format

Data stored in CoNLL-U format with standard 10-column structure. Includes token IDs, word forms, lemmas, POS tags (UPOS + German STTS), morphological features, and character offsets.

Documents delimited with `# newdoc id = <doc_id>` headers.

**Download**: File too large for GitHub. Available at: [Google Drive](https://drive.google.com/file/d/1bs7oI4dxBr2b7Hdp_BC9zWBHjSzHUCVl/view?usp=share_link)
A glimpse into the first 2000 lines can be found at data/processed/first2000.txt.

## Issues Identified

1. **Processing performance & limmitation**: Current throughput is ~10 docs/minute. Full corpus would take ~17 hours as some of the files are very large. For the sake of experimentation in this course, we limited the docs to 1000. 

2. **HTML artefacts**: HTML entities like `&quot;` appear in output instead of standard quotes.

3. **PDF-formatting**: Multiple consecutive newlines (e.g., `SpacesAfter=\s\n\s\n\s\n`) due to formatting from the original PDF-files. 

4. **Dependency parsing**: Not included in current version (columns 7-9 are placeholders).

Issues 1-3 are cosmetic and could be adressed in future steps if necessary. Issue 4 would require adding the `depparse` processor, approximately doubling processing time which is already quite high here.
All in all the file looks clean and ready to work with for future steps.

---



## Project Structure

```
NLP-Group-23/
├── data/
│   ├── processed/fincorpus_processed.conllu    # Milestone 1 output
│   ├── manual_annotation/
│   │   ├── sample_sentences.conllu             # Sampled
│   │   ├── sample_sentences_labeled.conllu     # Gold standard
│   │   └── labeling_guidelines.md              # Instructions
│   ├── sample_for_manual_annotation.py         # Sampling script
│   └── project_labeling.py                     # spaCy + regex labeling
├── evaluation/
│   └── compare_labeling_methods.py             # Comparison script
├── results/
│   ├── gpt_labeling_approach.py                # GPT labeling script (rule-based)
│   └── labeling_comparison/                    # Outputs
│       ├── chatgpt_predictions.tsv             # GPT predictions
│       └── project_predictions.conllu          # Project predictions
```

