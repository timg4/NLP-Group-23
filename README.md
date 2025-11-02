# NLP Project - Milestone 1

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

## Issues Identified

1. **Processing performance & limmitation**: Current throughput is ~10 docs/minute. Full corpus would take ~17 hours as some of the files are very large. For the sake of experimentation in this course, we limited the docs to 1000. 

2. **HTML artefacts**: HTML entities like `&quot;` appear in output instead of standard quotes.

3. **PDF-formatting**: Multiple consecutive newlines (e.g., `SpacesAfter=\s\n\s\n\s\n`) due to formatting from the original PDF-files. 

4. **Dependency parsing**: Not included in current version (columns 7-9 are placeholders).

Issues 1-3 are cosmetic and could be adressed in future steps if necessary. Issue 4 would require adding the `depparse` processor, approximately doubling processing time which is already quite high here.
All in all the file looks clean and ready to work with for future steps.

## Dependencies

```
python>=3.10,<3.13
nltk==3.9.1
stanza==1.8.2
datasets==3.0.1
conllu==4.5.3
regex==2024.9.11
torch==2.2.2
torchaudio==2.2.2
torchvision==0.17.2
tqdm==4.66.5
```
