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
- Get NER predictions
- Create `results/labeling_comparison/chatgpt_predictions.conllu` (same format as project predictions)

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
    --chatgpt results/labeling_comparison/chatgpt_predictions.conllu \
    --project results/labeling_comparison/project_predictions.conllu \
    --output results/labeling_comparison
```

---

## Entity Types

1. **ORGANIZATION (ORG)**: Companies, banks (e.g., "Deutsche Bank AG", "LBBW")
2. **MONETARY (MON)**: Currency amounts, percentages (e.g., "EUR 1000", "3,41 %")
3. **LEGAL_REFERENCE (LEG)**: Legal citations (e.g., "§ 15", "Art. 12 PVO")

---

## Results

After comparison, you'll get:
- `labeling_comparison.csv` - Comparison table
- `f1_comparison.png` - F1 scores chart
- `agreement_heatmap.png` - Cohen's Kappa visualization
- `detailed_report.txt` - Full analysis

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
└── results/labeling_comparison/                # Outputs
```

