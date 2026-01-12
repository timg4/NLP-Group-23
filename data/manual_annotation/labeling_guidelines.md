# Manual Labeling Guidelines for German Financial NER

## Overview

You will label **150 sentences** from German financial documents with three entity types using the BIO tagging scheme.

**Estimated time:** 1.5-2 hours

---

## Entity Types

### 1. ORGANIZATION (ORG)
**Companies, banks, financial institutions**

**What to include:**
- Full company names: "Landesbank Baden-Württemberg"
- Company abbreviations: "LBBW"
- Banks: "Deutsche Bank", "Sparkasse"
- Legal forms as part of name: "Deutsche Bank AG" (include "AG")
- Financial institutions: "Bundesbank", "EZB"

**What to exclude:**
- Generic terms: "die Bank", "eine Sparkasse" (without specific name)
- Person names (unless they're part of company name)

**Examples:**
```
Landesbank Baden-Württemberg → B-ORG I-ORG I-ORG
Deutsche Bank AG → B-ORG I-ORG I-ORG
LBBW → B-ORG
```

---

### 2. MONETARY (MON)
**Currency amounts, percentages, monetary values**

**What to include:**
- Amount + currency: "24.000.000 EUR", "100,00 €"
- Currency + amount: "EUR 500"
- Percentages: "3,41 %", "2,5%"
- Standalone monetary amounts in context: "5.000.000"

**What to exclude:**
- Dates: "2023.01.15"
- Phone numbers
- Pure numbers without monetary context: "100 Mitarbeiter"

**Examples:**
```
24.000.000 EUR → B-MON I-MON
3,41 % → B-MON I-MON
EUR 500.000 → B-MON I-MON
```

---

### 3. LEGAL_REFERENCE (LEG)
**Legal citations: paragraphs, articles, legal references**

**What to include:**
- Paragraph signs: "§ 15", "§ 15a"
- Articles: "Art. 12", "Art. 5 PVO"
- Absatz (paragraph): "Abs. 1", "Abs. 2"
- Number references in legal context: "Nr. 11", "Nr. 397"
- Combined: "§ 15 Abs. 2"
- Guidlines
- Policies

**What to exclude:**
- Page numbers: "Seite 5"
- Figure numbers: "Abbildung 3"
- Footnotes: "Fußnote 12"

**Examples:**
```
§ 15 → B-LEG I-LEG
Art. 12 PVO → B-LEG I-LEG I-LEG
Nr. 397 → B-LEG I-LEG
§ 15 Abs. 2 → B-LEG I-LEG I-LEG I-LEG
```

---

## BIO Tagging Scheme

### Tags:
- **B-ORG**: Beginning of organization entity
- **I-ORG**: Inside/continuation of organization entity
- **B-MON**: Beginning of monetary entity
- **I-MON**: Inside/continuation of monetary entity
- **B-LEG**: Beginning of legal reference entity
- **I-LEG**: Inside/continuation of legal reference entity
- **O**: Outside any entity (default for everything else)

### Rules:
1. **First token** of entity: Always use **B-** tag
2. **Continuation tokens**: Use **I-** tag
3. **Everything else**: Use **O**
4. **No gaps**: Don't skip tokens within an entity
5. **Adjacent entities**: Use B- for each new entity

---

## Examples

### Example 1: Organization
```
Sentence: "Die Deutsche Bank AG emittiert Anleihen."

TOKEN           NER_TAG
Die             O
Deutsche        B-ORG
Bank            I-ORG
AG              I-ORG
emittiert       O
Anleihen        O
.               O
```

### Example 2: Monetary
```
Sentence: "Der Betrag beläuft sich auf 24.000.000 EUR."

TOKEN           NER_TAG
Der             O
Betrag          O
beläuft         O
sich            O
auf             O
24.000.000      B-MON
EUR             I-MON
.               O
```

### Example 3: Legal Reference
```
Sentence: "Gemäß § 15 Abs. 2 ist dies zulässig."

TOKEN           NER_TAG
Gemäß           O
§               B-LEG
15              I-LEG
Abs.            I-LEG
2               I-LEG
ist             O
dies            O
zulässig        O
.               O
```

### Example 4: Multiple Entities
```
Sentence: "Die LBBW emittiert Anleihen im Wert von EUR 500.000 gemäß Art. 12."

TOKEN           NER_TAG
Die             O
LBBW            B-ORG
emittiert       O
Anleihen        O
im              O
Wert            O
von             O
EUR             B-MON
500.000         I-MON
gemäß           O
Art.            B-LEG
12              I-LEG
.               O
```

---

## Edge Cases & Difficult Situations

### Case 1: Punctuation within entities
**Include punctuation if it's part of the entity:**
```
Baden-Württemberg → B-ORG I-ORG I-ORG  (hyphen is token)
3,41 % → B-MON I-MON I-MON  (comma and percent included)
```

### Case 2: Multiple entities of same type adjacent
**Use B- for the second entity:**
```
"Deutsche Bank and Sparkasse"
Deutsche    B-ORG
Bank        I-ORG
and         O
Sparkasse   B-ORG    ← NEW entity, use B- again
```

### Case 3: Abbreviations
**Include dots if they're part of the token:**
```
Art. 12 → B-LEG I-LEG  (if "Art." is one token)
Nr . 11 → B-LEG I-LEG I-LEG  (if "." is separate token)
```

### Case 4: Uncertain cases
**When in doubt:**
- Ask yourself: "Is this important for financial/legal understanding?"
- ORG: Does it refer to a specific organization?
- MON: Does it represent a monetary value or percentage?
- LEG: Does it cite a law, paragraph, or article?
- If still unsure, mark as **O** and make a note

---

## Workflow in Excel/Google Sheets

### Step-by-step:

1. **Open `sample_sentences.tsv`** in Excel or Google Sheets

2. **You'll see these columns:**
   - `sentence_id`: Sentence number
   - `token_id`: Token position in sentence
   - `token`: The actual word
   - `lemma`: Base form
   - `upos`: Part-of-speech tag
   - `xpos`: Detailed POS tag
   - `ner_tag`: **← FILL THIS COLUMN**

3. **For each token, enter the appropriate BIO tag:**
   - Type: `B-ORG`, `I-ORG`, `B-MON`, `I-MON`, `B-LEG`, `I-LEG`, or `O`
   - Use autocomplete after typing first few tags

4. **Use POS tags to help:**
   - `PROPN` (proper noun) → likely ORG
   - `NUM` (number) → check if MON
   - Check context of surrounding words

5. **Take breaks!**
   - Do 30-50 sentences, then take a 10-minute break
   - Don't rush - quality over speed

6. **Save frequently** as you work

7. **When done, save as `sample_sentences_labeled.tsv`**

---

## Quality Checks

Before submitting, check:

- ✅ Every row has a `ner_tag` (no empty cells)
- ✅ All B- tags have corresponding I- tags (if multi-word entity)
- ✅ No I- tag appears without a preceding B- tag
- ✅ Tags are spelled correctly (case-sensitive!)
- ✅ Entity boundaries make sense (read the full entity)

---

## Tips for Faster Labeling

1. **Start with O everywhere** - then update entities
2. **Use Excel filters** - filter by `upos=PROPN` to find ORG candidates
3. **Search for keywords:**
   - Search "EUR", "€", "%" → check for MON
   - Search "§", "Art.", "Nr." → mark as LEG
4. **Copy-paste tags** for repeated entities
5. **Focus on context** - read the full sentence, not just individual words

---

## Common Mistakes to Avoid

❌ **Wrong:**
```
Deutsche Bank AG → B-ORG B-ORG B-ORG  (should use I- for continuation)
```

✅ **Correct:**
```
Deutsche Bank AG → B-ORG I-ORG I-ORG
```

---

❌ **Wrong:**
```
EUR 500.000 → B-MON O I-MON  (gap in entity)
```

✅ **Correct:**
```
EUR 500.000 → B-MON I-MON
```

---

❌ **Wrong:**
```
§ 15 Abs. 2 → B-LEG I-LEG O I-LEG  (gap for "Abs.")
```

✅ **Correct:**
```
§ 15 Abs. 2 → B-LEG I-LEG I-LEG I-LEG  (all part of same legal reference)
```

---

## Questions?

If you encounter difficult cases:
1. Make your best judgment
2. Add a note/comment in a separate column
3. Be consistent - use the same rule for similar cases

---

## Final Checklist

Before you finish:
- [ ] All 150 sentences labeled
- [ ] No empty `ner_tag` cells
- [ ] Quality checks passed
- [ ] File saved as `sample_sentences_labeled.tsv`
- [ ] Ready for comparison!

**Thank you for your careful work! 🎉**

Your manual labels will be the **gold standard** for evaluating automated methods.
