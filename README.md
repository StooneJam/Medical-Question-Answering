# Medical Question Answering (PubMedQA)

This project classifies each PubMedQA question as `yes`, `no`, or `maybe`, using the expert
labelled PQA-L subset. The work starts from a simple baseline and then adds four improvements,
and it finishes with a short study on how well answer quality can be predicted from question
features. The folders are numbered so that they follow the order of the final report, from the
introduction through to the evaluation.

## Project structure

The repository is organised into numbered folders. Folder `01` and `02` cover the data and the
way it is prepared. Folder `03` holds the baseline model, and folders `04` to `07` hold the four
improvements in the order the report discusses them. Folder `08` covers the answer quality
prediction study. The remaining folders hold shared code, the report, and supporting material.

| Folder | Report section | Contents |
|--------|----------------|----------|
| `01_EDA/` | Introduction | Exploratory data analysis and the human performance reference |
| `02_Preprocessing/` | Methods, setup | Dataset splitting into ten cross-validation folds and a test set |
| `03_Baseline_TFIDF-LR/` | Methods, baseline system | TF-IDF features with logistic regression, and the error analysis |
| `04_Exp1_BiomedBERT-LR/` | Improvement 1 | Frozen BiomedBERT embeddings with logistic regression |
| `05_Exp2_BioClinicalModernBERT/` | Improvement 2 | End to end BioClinical ModernBERT classification |
| `06_Exp3_BioGPT/` | Improvement 3 | BioGPT generative question answering, zero shot and fine tuned |
| `07_Exp4_BioMed-R1-8B/` | Improvement 4 | BioMed-R1-8B reasoning model, zero shot and fine tuned |
| `08_AnswerQualityPrediction/` | Evaluation, answer quality prediction | Regression on question difficulty across the model outputs |
| `utils/` | Shared code | Helper modules for evaluation, preprocessing, and LLM calls |
| `Overleaf_Report/` | Report source | LaTeX source of the final report, kept in sync with Overleaf |
| `report/` | Report | Report drafts and the compiled final PDF |
| `docs/research/` | Supporting material | Reference papers |
| `docs/project-management/` | Supporting material | Meeting notes and the project timeline |

Each experiment folder keeps its own copy of the data, which holds `ori_pqal.json`,
`test_set.json`, `test_ground_truth.json`, and the ten folds `pqal_fold0` to `pqal_fold9`. The
notebooks read the data using paths that are relative to their own folder, so the `data` folder
should stay next to each notebook.

## Experiment pipeline

The work follows one pipeline and changes two parts of it at a time, the text representation and
the prediction method. Each row below adds more capacity than the row above it, which lets us
measure what each change is worth.

| Stage | Text representation | Prediction method | Model |
|-------|---------------------|-------------------|-------|
| Baseline | TF-IDF | Linear classification | Logistic regression |
| Experiment 1 | BiomedBERT embeddings | Linear classification | Logistic regression |
| Experiment 2 | Contextual transformer | End to end classification | BioClinical ModernBERT |
| Experiment 3 | Generative biomedical | Answer generation | BioGPT |
| Experiment 4 | Generative biomedical | Answer generation with reasoning | BioMed-R1-8B |

## Tech stack

The code is written in Python and runs inside Jupyter notebooks. The classical models use
scikit-learn for TF-IDF features, logistic regression, and the evaluation metrics. The
transformer and generative models use PyTorch with the Hugging Face `transformers` and
`datasets` libraries. The larger models are adapted with PEFT and TRL, which allow low rank
fine tuning without training every weight. Data handling and figures use pandas, NumPy,
Matplotlib, seaborn, and SciPy, and spaCy is used for some of the question feature extraction.
The final report is written in LaTeX.

## Directory tree

```
Medical-Question-Answering/
├── README.md
├── utils/                              # evaluation.py, preprocess.py, llm_utils.py
│
├── 01_EDA/                             # EDA and human performance reference
├── 02_Preprocessing/                  # dataset splitting and preprocessing
├── 03_Baseline_TFIDF-LR/              # TF-IDF with logistic regression, error analysis
├── 04_Exp1_BiomedBERT-LR/            # frozen BiomedBERT with logistic regression
├── 05_Exp2_BioClinicalModernBERT/    # end to end BioClinical ModernBERT
├── 06_Exp3_BioGPT/                   # BioGPT generative QA
├── 07_Exp4_BioMed-R1-8B/            # BioMed-R1-8B reasoning model
├── 08_AnswerQualityPrediction/       # question difficulty regression
│
├── report/                            # report drafts and compiled Group_21.pdf
├── Overleaf_Report/                   # LaTeX source, kept in sync with Overleaf
└── docs/
    ├── research/                      # reference papers
    └── project-management/            # meeting notes and timeline
```
