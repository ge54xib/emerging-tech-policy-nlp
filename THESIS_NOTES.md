# Thesis Writing Notes — Methodology Chapter

## 3.1 Research Design

- Follows **Design Science Research (DSR)** paradigm (Hevner et al. 2004; Peffers et al. 2007)
- Artifact type: **analytical NLP pipeline** — a computational instrument, not a software product
- DSR is appropriate because the pipeline is constructed to solve a concrete research problem (scalable Quadruple Helix analysis) and evaluated against design objectives
- Research method combines: corpus construction → artifact development → technical evaluation (F1) → utility evaluation (RQ answers)
- Iterative design: pipeline steps were refined based on intermediate outputs (e.g. entity-in-sentence filtering, sentence splitter fixes, class imbalance in annotation)

---

## 3.2 Problem Identification and Motivation

- National quantum strategies are policy documents that define how governments position themselves in the global quantum race — but no systematic, cross-national comparison of actor structures exists
- The **Triple Helix** (Ranga & Etzkowitz 2013) and its **Quadruple Helix** extension (Carayannis & Campbell 2009) provide the theoretical lens: academia, industry, government, and civil society as innovation spheres
- Manual analysis does not scale: 23 documents, 16 countries, 2018–2025, varying length and structure
- Existing NLP approaches to policy analysis focus on topic modelling or sentiment — not actor-relation extraction grounded in an innovation systems framework
- Gap: no pipeline exists that operationalises Triple Helix concepts (components, relationships, spaces) computationally on policy text

---

## 3.3 Definition of the Solution Objectives

The pipeline must:
1. **Extract** named entities (actors) from policy PDFs reliably
2. **Classify** each actor into the Quadruple Helix taxonomy (government / academia / industry / civil_society / intermediary)
3. **Identify co-occurring actor pairs** within paragraphs as candidate relations
4. **Classify relations** between actor pairs into the 5 TH relation types (Ranga & Etzkowitz 2013)
5. **Map TH innovation spaces** (knowledge / innovation / consensus / public) to sentences
6. **Aggregate** actor prominence and interaction patterns into country-level profiles
7. **Answer RQ1–RQ3** with reproducible outputs (JSON, figures, CSV)

Design constraints:
- Zero-shot or few-shot only — no large labelled training corpus available
- Must run on policy text in English (varying document quality, PDF artefacts)
- Reproducible: all steps configurable via `.env`, all outputs versioned

---

## 3.4.1 Corpus Construction

- **22 processed documents** (23 PDFs; DNK split into 2 parts) from **16 countries** (AUS, CAN, CZE, DNK, ESP, FIN, FRA, GBR, GER, IRL, ITA, JPN, KOR, NLD, SVN, USA), published **2018–2025**

| Country | Year | Document |
|---|---|---|
| AUS | 2023 | National Quantum Strategy |
| CAN | 2022 | Canada's National Quantum Strategy |
| CZE | 2025 | National Strategy for Quantum Technologies |
| DNK | 2023 | Strategy for Quantum Technology (Part 1 + 2) |
| ESP | 2025 | Strategy of Quantum Technologies of Spain |
| FIN | 2025 | Finland's Quantum Technology Strategy 2025–2035 |
| FRA | 2021 | National Strategy on Quantum Technologies |
| GBR | 2023 | National Quantum Strategy |
| GER | 2018 | Quantum technologies – from basic research to market |
| GER | 2023 | Quantum Technologies Conceptual Framework Programme |
| IRL | 2023 | Quantum 2030 |
| ITA | 2025 | Italian Strategy for Quantum Technologies |
| JPN | 2022 | Vision of Quantum Future Society |
| JPN | 2023 | Strategy of Quantum Future Industry Development |
| JPN | 2024 | Promotion Measures for the Development of Quantum Industries |
| KOR | 2023 | Korea's National Quantum Strategy |
| NLD | 2019 | National Agenda for Quantum Technology |
| SVN | 2025 | Strategy for the Development of Quantum Technologies |
| USA | 2018 | National Strategic Overview for Quantum Information Science |
| USA | 2020 | A Strategic Vision for America's Quantum Networks |
| USA | 2022 | QIS Workforce Development National Strategic Plan |
- Collected from official government portals and ministerial websites; English originals or official translations
- Naming convention: `<COUNTRY>_<YEAR>_<Title>.pdf`
- Corpus spans the full policy lifecycle: early visions (GER 2018, NLD 2019) through mature strategies (FIN 2025, ITA 2025)
- Document lengths vary substantially — capped at 300,000 characters for NER processing (`MAX_DOC_CHARS`)
- No pre-existing labelled corpus for this domain → motivates few-shot and zero-shot approaches throughout

---

## 3.4.2 Corpus Preprocessing

- **Tool:** Adobe PDF Services API (Extract API) — structured extraction preserving paragraph and heading tags
- Extracts structured JSON with role tags (`P`, `H1`–`H6`, `TOC`, `TOCI`, `FOOTNOTE`, `L`, `LI`, etc.)
- Excluded tags: `TOC`, `TOCI`, `FOOTNOTE` (configurable via `PREPROCESS_EXCLUDE_PATH_TAGS`)
- Page exclusions at three levels: global, country-level, document-level (via `.env` and `configs/step0_exclude_pages_by_doc.template.txt`)
- Output: plain text per document (`data/processed/step0/text/`) and structured JSON (`step0/structured/`)
- Preprocessing removes cover pages, tables of contents, reference lists, and appendices that would introduce noise into NER
- Paragraph boundaries preserved — critical for co-occurrence extraction in Step 3

---

## 3.4.3 Components: Actor Identification and Classification

### Named Entity Recognition (Step 1)
- Model: **Flair `ner-english-large`** (Akbik et al. 2019) — BiLSTM-CRF with Flair string embeddings
- Extracts `ORG` (institutional actors) and `PER` (individual actors) entities
- Batch inference: `FLAIR_NER_BATCH_SIZE=32`, GPU recommended
- Output: `data/processed/step2/entities_ner.jsonl` with span offsets, confidence scores, document provenance

### Quadruple Helix Classification (Step 2)
- **Manual classification** of unique entity labels into 5-level taxonomy:

| Level | Field | Values |
|---|---|---|
| 1 | `level_1_actor_type` | institutional / individual |
| 2 | `level_2_sphere_boundary` | single-sphere / multi-sphere |
| 3 | `level_3_exact_category` | fine-grained (e.g. "national government institutions") |
| 4 | `level_4_innovation_type` | R&D / Non R&D / Both |
| 5 | `level_5_helix` | government / industry / academia / civil_society / intermediary |

- Stored in `data/processed/step2/manual_classification_labels.json`
- Non-entity statuses filtered: `historical_individual`, `ner_error`, `not_actor`, `not_specific`, `non_entity_phrase`, `unclear`

### Corpus statistics (after classification):
- **5,077 classified entity mentions** across 22 documents
- Helix distribution:

| Helix | Mentions | Share |
|---|---|---|
| Government | 2,896 | 57.1% |
| Academia | 1,031 | 20.3% |
| Intermediary | 624 | 12.3% |
| Industry | 422 | 8.3% |
| Civil Society | 88 | 1.7% |

- Strong government dominance — itself a finding for RQ1/RQ3
- Civil society marginally represented (88 mentions) — key finding for QH extension
- `intermediary` captures hybrid actors (tech transfer offices, science parks, PPPs) that span spheres — consistent with R&E 2013's multi-sphere institutions

### Theoretical grounding:
- Ranga & Etzkowitz (2013) — three actor distinctions: individual/institutional, single/multi-sphere, R&D/non-R&D
- Fourth helix (civil society) added following Carayannis & Campbell (2009) — "Mode 3" knowledge production involving media, culture, public legitimation

---

## 3.4.4 Relationships: Relation Extraction and Classification

### Co-occurrence Extraction (Step 3)
- Entity pairs extracted within **paragraph windows** — two entities appearing in the same paragraph are candidate relations
- Central sentence (`central_sent_text`): the sentence containing the entity mention pair — used as NLI premise
- Only **cross-helix pairs** retained (h1 ≠ h2) — same-helix pairs not relevant to RQs
- **12,899 cross-helix co-occurrence pairs** across all documents (after filtering same-helix pairs and requiring sentence text)
- Deduplication by `(doc_id, paragraph_id, sentence_id)`

### Relation Classification — Baseline (Step 3 NLI)
- **Model:** `roberta-large-mnli` (Liu et al. 2019) in fp16, zero-shot NLI
- 5 relation types from Ranga & Etzkowitz (2013): technology_transfer, collaborative_leadership, substitution, networking, collaboration_conflict_moderation
- 4 hypothesis templates × 2 directions per relation = **40 NLI calls per pair**
- Threshold: `NLI_THRESHOLD=0.5` on entailment score
- Fallback: `no_explicit_relation` when no template exceeds threshold

### Relation Classification Experiments (Evaluation)
Gold standard: **100 manually annotated pairs** (`evaluation/annotation.json`) — stratified by helix-pair bucket, annotated using `evaluation/codebook.md`

Five alternative approaches evaluated:

| ID | Method | Paper | Key idea |
|---|---|---|---|
| R1 | NLI Pipeline (baseline) | Lewis et al. 2019 | roberta-large-mnli, label as hypothesis |
| R2 | NLI-Sainz | Sainz et al. 2021 | Entity-marked premise [E1]…[E2], full-sentence hypotheses |
| R3 | Claude CoT | Kojima et al. 2022 | Chain-of-thought prompting with claude-sonnet-4-6 |
| R4 | GoLLIE Guidelines | Sainz et al. 2023 | Class definitions as Python docstrings, GPT-4o |
| R5 | GPT-RE | Wan et al. 2023 | Entity-aware SimCSE demo retrieval + gold label-induced reasoning, GPT-4o |
| R6 | Self-Consistency | Wang et al. 2023 | k=10 samples, majority vote, claude-haiku |

Annotation distribution: networking (55), no_explicit_relation (36), collaborative_leadership (13), technology_transfer (3) — substitution and collaboration_conflict_moderation absent from corpus sample → finding in itself

---

## 3.4.5 Functions: Knowledge, Innovation, and Consensus Spaces Mapping

### TH Space Definition (Ranga & Etzkowitz 2013)
Three functional spaces + public space (Carayannis & Campbell 2009):

| Space | Activity |
|---|---|
| Knowledge space | R&D, scientific collaboration, education, building knowledge base |
| Innovation space | Tech transfer, commercialisation, spin-offs, IP, incubators |
| Consensus space | Governance, policy dialogue, strategy, regulation, coordination |
| Public space | Public trust, ethics, equity, media, culture, gender diversity, civic engagement |

### Static Mapping (Primary — used for RQ2/RQ3)
- Theoretically grounded helix-pair → space mapping (`TH_SPACE_MAP` in `rq2.py`)
- Deterministic, no model uncertainty
- academia–government → knowledge_space; academia/industry/intermediary triads → innovation_space; government–industry/intermediary → consensus_space; any civil_society pair → public_space

### Space Classification Experiments (Evaluation)
Training data: **92 manually annotated sentences** (`Experiments/Spaces/spaces_labels.json`) — stratified by helix-pair, annotated using `evaluation/codebook_spaces.md`
Gold standard: **100 manually annotated sentences** (`evaluation/annotation_spaces.json`)

Five approaches evaluated:

| ID | Method | Paper | Key idea |
|---|---|---|---|
| S1 | Helix-pair baseline | — | Static TH_SPACE_MAP, no model |
| S2 | NLI Pipeline | Lewis et al. 2019 | roberta-large-mnli, space label as hypothesis |
| S3 | INSTRUCTOR embeddings | Su et al. 2023 | Task-instruction-aware embeddings, prototype classification |
| S4 | SetFit | Tunstall et al. 2022 | Few-shot contrastive fine-tuning of sentence-transformer |
| S5 | GPT-4 few-shot | Brown et al. 2020 | In-context learning, 3 demos per class |
| S6 | LLM Augmentation + FastFit | Truveta 2025 | GPT-4 synthetic data for minority class (public_space) + FastFit |

---

## 3.4.6 Strategy-Level Profiles

- **Helix Balance Index (HBI):** entropy-based measure of actor distribution across helices
  - HBI = 1.0 → perfectly balanced; HBI → 0 → dominated by one helix
- **Configuration classification** (Etzkowitz & Leydesdorff 2000):
  - Balanced: HBI ≥ 0.80
  - Statist: HBI < 0.80 AND government share > 50%
  - Laissez-faire: HBI < 0.80 AND industry leads non-government helices
  - Mixed: HBI < 0.80 AND academia or civil_society leads
- Outputs: per-country actor shares, HBI, configuration label, helix-pair interaction heatmap

---

## 3.5 Demonstration

### Worked Example — Single Sentence Walkthrough

Script: `Experiments/demo_pipeline.py` — run with `python Experiments/demo_pipeline.py`

**Chosen sentence** (CAN_2022 — Canada's National Quantum Strategy):
> *"Over the past 10 years, the Government of Canada has provided $51 million to support the IQC."*

Chosen because: short, unambiguous entities from two different helices, clear funding/support relation, obvious knowledge-space activity.

---

**Step 1 — Input Sentence**
- Raw text from preprocessed PDF (`data/processed/step0/text/CAN_2022_*.txt`)
- No modification — exactly as extracted by Adobe PDF Services

---

**Step 2 — NER + Helix Classification**

| Span | NER label | Helix | Category |
|---|---|---|---|
| `Government of Canada` | ORG | government | national government institutions |
| `IQC` | ORG | academia | public research organisations |

- NER by Flair `ner-english-large` (Akbik et al. 2019) — detects both spans as `ORG`
- Helix assignment from `manual_classification_labels.json` (Step 2 human annotation)
- The sentence may contain more entities (e.g. monetary amounts filtered as non-actors); only cross-helix ORG/PER entities proceed
- Pair selected: `Government of Canada` [government] ↔ `IQC` [academia]

---

**Step 3 — Relation Extraction (NLI)**

Premise fed to NLI: entity-masked version — *"[GOV] has provided $51 million to support [ACAD]."*

NLI model scores entailment probability for each of 5 hypothesis templates:

| Relation | Hypothesis | Confidence |
|---|---|---|
| **technology_transfer** | *[GOV] transfers research results or funding to [ACAD] for application.* | **0.81** ← chosen |
| collaborative_leadership | *[GOV] leads and coordinates [ACAD] within a cross-sector initiative.* | 0.43 |
| networking | *[GOV] and [ACAD] are joint members of a consortium or network.* | 0.21 |
| substitution | *[GOV] fills a gap left by [ACAD]'s absence.* | 0.04 |
| collaboration_conflict_moderation | *[GOV] helps resolve tensions involving [ACAD].* | 0.02 |

→ **Assigned: `technology_transfer`** (highest entailment score, above threshold 0.5)

Model: `cross-encoder/nli-deberta-v3-large` (He et al. 2021)

---

**Step 4 — TH Space Classification (NLI)**

Full sentence used as premise (not masked) — space is a property of the sentence context, not the entity pair.

| Space | Hypothesis | Confidence |
|---|---|---|
| **knowledge_space** | *This sentence describes knowledge generation, R&D resources, or education.* | **0.79** ← chosen |
| consensus_space | *This sentence describes governance, policy dialogue, or strategy.* | 0.38 |
| innovation_space | *This sentence describes tech transfer, commercialisation, or IP.* | 0.22 |
| public_space | *This sentence describes public engagement, ethics, or equity.* | 0.03 |

→ **Assigned: `knowledge_space`** (highest entailment score)

Note: in RQ2/RQ3 the primary space assignment uses the static helix-pair mapping (government ↔ academia → `knowledge_space`), which agrees with the NLI result here.

---

**Summary of this example:**

```
Sentence  : "Over the past 10 years, the Government of Canada has
             provided $51 million to support the IQC."
Entity 1  : Government of Canada  [government]
Entity 2  : IQC                   [academia]
Relation  : technology_transfer   (conf. 0.81)
TH Space  : knowledge_space       (conf. 0.79)
```

This pair contributes to:
- **RQ1**: government and academia actor counts for Canada
- **RQ2**: government–academia helix-pair interaction, knowledge_space count
- **RQ3**: Canada's helix balance index and configuration profile

- Pipeline run on all 22 documents end-to-end
- RQ1: actor prominence per country → helix shares, HBI, configuration classification
- RQ2: helix-pair interaction intensity → co-occurrence counts by pair, TH space distribution
- RQ3: country-level QH system type profiles → radar charts, configuration matrix
- All outputs in `outputs/rq1/`, `outputs/rq2/`, `outputs/rq3/`

---

## 3.6.1 Technical Evaluation

### Relation Extraction Results (100-entry gold standard, `annotation.json`)

Eval set distribution: networking 55 · no_explicit_relation 36 · collaborative_leadership 13 · technology_transfer 3 · substitution 0 · collaboration_conflict_moderation 0

| Method | Accuracy | Macro F1 | Weighted F1 | Notes |
|---|---|---|---|---|
| R1 NLI Pipeline | 0.490 | 0.171 | 0.422 | Zero-shot baseline |
| R2 NLI-Sainz | 0.440 | 0.114 | 0.323 | Entity-marked premise |
| R3 Claude CoT | 0.490 | 0.273 | 0.536 | claude-sonnet-4-6 |
| R4 GoLLIE Guidelines | **0.610** | **0.306** | **0.646** | GPT-4o, best overall |
| R5 GPT-RE | — | — | — | Results pending |
| R6 Self-Consistency | — | — | — | Results pending |

- Zero F1 on `substitution` and `collaboration_conflict_moderation` across all methods — absent from eval set; finding in itself (rare in quantum strategies)
- `collaborative_leadership` F1 ranges 0.28–0.31 — consistently identified by LLM-based methods

### Space Classification Results (100-entry gold standard, `annotation_spaces.json`)

Eval set distribution: knowledge_space 46 · consensus_space 26 · innovation_space 19 · no_explicit_space 10 · public_space 0

| Method | Accuracy | Macro F1 | Weighted F1 | Notes |
|---|---|---|---|---|
| S1 Helix-pair baseline | 0.260 | 0.181 | 0.258 | Deterministic mapping |
| S2 NLI Pipeline | 0.320 | 0.150 | 0.307 | knowledge_space bias |
| S3 INSTRUCTOR | — | — | — | Results pending |
| S4 SetFit | **0.700** | 0.490 | 0.696 | 92 training examples |
| S5 GPT-4 few-shot | 0.720 | **0.545** | **0.714** | 3 demos per class |
| S6 LLM Augmenter | — | — | — | Results pending |

- SetFit achieves near-GPT-4 performance with no API cost and only 92 training sentences
- Zero F1 on `public_space` — absent from eval set; civil society marginally represented

---

## 3.6.2 Utility Evaluation

- Does the pipeline answer the RQs with interpretable, theoretically grounded outputs?
- RQ1: actor distribution reveals government dominance in all strategies → statist configurations predominant
- RQ2: networking is the dominant relation type; technology_transfer and substitution are rare → quantum strategies emphasise coordination over concrete knowledge exchange
- RQ3: country profiles reveal variation between balanced (NLD, FIN) and statist (KOR, CHN-adjacent) configurations
- Limitations to address: civil society underrepresentation (only 88 entities), NLI threshold sensitivity, PDF quality variation

---

## Key Papers to Cite

| Section | Paper |
|---|---|
| Framework | Ranga & Etzkowitz (2013) — Triple Helix systems |
| Framework | Carayannis & Campbell (2009) — Mode 3, Quadruple Helix |
| Framework | Etzkowitz & Leydesdorff (2000) — helix configurations |
| DSR | Hevner et al. (2004) — Design Science in IS |
| DSR | Peffers et al. (2007) — DSR methodology |
| NER | Akbik et al. (2019) — Flair NER |
| NLI baseline | Liu et al. (2019) — RoBERTa |
| NLI zero-shot RE | Lewis et al. (2019) / Yin et al. (2019) — NLI for classification |
| NLI-Sainz | Sainz et al. (2021) — textual entailment for RE |
| GoLLIE | Sainz et al. (2023) — annotation guidelines as code |
| GPT-RE | Wan et al. (2023) — GPT-RE, EMNLP 2023 |
| Self-Consistency | Wang et al. (2023) — self-consistency CoT |
| CoT | Kojima et al. (2022) — zero-shot CoT |
| SetFit | Tunstall et al. (2022) — SetFit few-shot |
| INSTRUCTOR | Su et al. (2023) — INSTRUCTOR embeddings, ACL 2023 |
| LLM Augmentation | Truveta Research (2025) — structured LLM augmentation |
| GPT few-shot | Brown et al. (2020) — GPT-3, in-context learning |
