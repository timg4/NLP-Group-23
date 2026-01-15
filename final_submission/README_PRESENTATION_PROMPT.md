# Prompt for ChatGPT (PowerPoint / Final Presentation)

Copy-paste everything below into ChatGPT. Then attach / paste the referenced result files when asked.

---

You are helping me create a final PowerPoint presentation for a German financial/legal NER project.

## Goal
Create a slide-by-slide outline (titles + 3–6 bullets per slide) that explains:
- the task and labels (ORG, MON, LEG)
- the dataset + annotation format
- the evaluation setup (shared stratified 80/20 split, seed = 2323)
- compared methods + why they matter
- results (token-level metrics AND overlap/span-level metrics)
- runtime comparison
- qualitative error patterns + limitations
- a short conclusion + future work

Keep it concise, clear, and presentation-ready. Assume the audience is an NLP class / project evaluation panel.

## Project structure (what matters for the presentation)
- Data: `data/manual_annotation/hand_labelled.conllu`
- Evaluation runners (each creates its own results folder):
  - RuleChef: `final_submission/run_rulechef.py`
- OpenAI direct labeling: `final_submission/run_openai_ner.py`
- Enhanced rule-based baseline: `final_submission/run_enhanced_rule_based.py`
- CRF baseline: `final_submission/run_crf.py`
- German BERT token classification: `final_submission/run_bert_ner.py`
- spaCy labeling (spaCy + regex): `final_submission/run_project_labeling.py`
  - Milestone2 Naive Bayes (uniform priors): `final_submission/run_milestone2_nb_uniform_priors.py`
- Shared evaluation logic:
  - Stratified split helper (seeded): `final_submission/utilities/stratified_split.py`
  - Metrics + overlap evaluation: `final_submission/utilities/common.py`
- Summary aggregation:
  - `final_submission/utilities/summarize_results.py` produces `final_submission/results/summary.txt`

## What files I will provide you (you should ask me to paste them)
1) `final_submission/results/summary.txt`
2) For each method folder under `final_submission/results/<ModelName>/metrics_summary.txt`
   (at least: RuleChef, OpenAI_NER, EnhancedRuleBasedNER, CRF, SpacyLabeling, TokenNB_uniform_priors, BERT)

## Important evaluation details (must be reflected accurately)
- Split: stratified 80/20 by sentence-level presence of entity types, seed = 2323
- Token-level metrics: labels collapsed to {ORG, MON, LEG, O} before classification report
- Overlap metrics: span-level overlap precision/recall/F1 per label + micro (more boundary-tolerant)
- Runtime: each runner reports seconds in its metrics output

## Requested slide outline
Give me ~10–14 slides in this order (adjust if needed):
1. Title / Team / Problem statement
2. Task definition (NER labels + examples)
3. Dataset + annotation (CoNLL-U, hand-labeled) + label distribution (if available)
4. Methods overview (bullet list of all compared approaches)
5. Evaluation design (split, seed, token vs overlap metrics)
6. Results table (from `summary.txt`) + key ranking
7. Per-label results (ORG/MON/LEG): what works best for which label
8. Overlap vs exact: explain why overlap is informative + highlight deltas
9. Runtime comparison + practicality tradeoffs
10. Error analysis: common FP/FN patterns (use metrics notes; ask me for examples if needed)
11. Discussion: strengths/weaknesses of each method
12. Conclusion (what we learned) + recommendation
13. Future work (next steps)
14. Backup slides (optional): extra tables / examples

## Output format
- For each slide: `Slide N: Title` then bullets
- Provide a short “speaker notes” paragraph after each slide
- End with a 5-sentence executive summary I can read out loud

Before you start writing slides, ask me to paste:
- `final_submission/results/summary.txt`
- and the per-model `metrics_summary.txt` files you need.

---
