# EXP1-report

# 1. Experimental Evaluation

### 1.1 Performance Metrics and Their Limitations

We evaluate all models using two metrics: accuracy and macro-averaged F1 score. Accuracy measures the proportion of correctly classified instances across all three labels. However, accuracy alone can be misleading when class distributions are uneven, as a model biased toward the majority class may score high without making useful predictions. We therefore treat macro-F1 as our primary metric, as it averages F1 equally across all three classes regardless of frequency, and penalises models that fail to recognise minority classes. 

A limitation of macro-F1 is that it treats failure on each class as equally important regardless of its real-world significance, which may not reflect actual clinical priorities. For the purpose of evaluating whether models can genuinely distinguish all three answer categories, however, macro-F1 remains the more appropriate choice.

### 1.2 Experimental Setup

All experiments use the expert-labelled subset of PubMedQA (PQA-L), which contains 1,000 instances. We follow the standard evaluation protocol: 10-fold cross-validation on a fixed 500-instance split, with the remaining 500 instances reserved as a held-out test set. 

In Experiment 1, the frozen BiomedBERT model was used to encode the question and context separately. This model was pre-trained on PubMed abstracts and full-text articles, making its representations well-suited to biomedical language. The questions were limited to 128 tokens and the contexts to 384 tokens (MAX-Q and MAX-C respectively), for a combined total within BiomedBERT’s 512-token limit. The output of all tokens was average-pooled to obtain a 768-dimensional vector for each input. We then test four vector combination methods to merge the question and context embeddings: 

- Concat: Concatenation, preserving both representations independently.
- Diff: Element-wise subtraction, capturing semantic distance between question and context.
- Prod: Element-wise multiplication, highlighting dimensions where both vectors agree
- Concat-Diff: Concatenation of both the original vectors and their difference.

Before training, all features are standardised. We then apply a logistic regression classifier with the same settings as the baseline.

We tested seven parameter configurations in total, varying the following factors: token length allocation (MAX-Q, MAX-C), pooling method, regularisation strength C, and context length limit (CTX-LIMIT). The best-performing configuration is reported as the main result.

All experiments were completed on a NVIDIA T4 GPU. Encoding all 10 folds took approximately 130 seconds in total, and training logistic regression on each fold took less than 0.5 seconds.

### 1.3 Results

Table 1: 10-Fold Cross-Validation Results

| Model | Combination | Accuracy | Macro-F1 | F1 no | F1 yes | F1 maybe |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + LR (Baseline) | n/a | 0.5160 | 0.353 | 0.339 | 0.643 | 0.097 |
| BiomedBERT + LR | concat | 0.5280 | 0.4195 | 0.4613 | 0.6418 | 0.1554 |
| BiomedBERT + LR | diff | 0.5320 | 0.4379 | 0.4624 | 0.6369 | 0.2143 |
| BiomedBERT + LR | prod | 0.4960 | 0.3831 | 0.3933 | 0.6229 | 0.1329 |
| BiomedBERT + LR | concat-diff | 0.5280 | 0.4228 | 0.4725 | 0.6398 | 0.1561 |

The best result was achieved using the diff combination method, with a macro-F1 score of 0.4379, which is 24.1% higher than the baseline (0.353). Both accuracy and macro-F1 improved over the baseline, with accuracy rising from 0.516 to 0.532. 

Table 2 presents the parameter sensitivity analysis. The diff combination achieved the best or near-best macro-F1 in five of the seven configurations.

- Config 2 (MAX-Q=64, MAX-C=448)

Reallocates the token budget to give more capacity to the context at the cost of the question. Performance drops slightly (macro-F1: 0.4168), suggesting that truncating the question to 64 tokens loses important information, and that the default allocation is already well-balanced.

- Config 3 (CLS pooling)

Uses only the CLS token representation rather than averaging all tokens. The drop in macro-F1 from 0.4379 to 0.4061 indicates that the CLS token alone does not capture sufficient information from biomedical text, and that mean pooling over all tokens is a better summary strategy.

- Config 4 (C=100)

Increases the logistic regression regularisation strength from C=10 to C=100, allowing the model to fit more tightly to the training data. The slight decline (macro-F1: 0.4219) suggests the default C=10 strikes a better bias–variance trade-off given the small training set.

- Config 5 (CTX-LIMIT=1)

Restricts the context to only the first paragraph of the abstract. This causes the largest drop (macro-F1: 0.3541), confirming that the model depends on information spread across multiple paragraphs to form a reliable answer.

- Config 6 (CTX-LIMIT=5)

Limits the context to the first five paragraphs, which retains most of the useful content. Performance (macro-F1: 0.4325) is close to the best configuration, indicating that the first five paragraphs cover most of the relevant information. But 3 is still the best option.

- Config 7 (CTX-LIMIT=None)

Imposes no paragraph limit, relying solely on the 384-token truncation. Performance (macro-F1: 0.4332) is comparable to Config 6, suggesting that most informative content falls within the token budget anyway.

Table 2: Parameter Sensitivity Analysis (best combination mode per configuration)

| Config | Key change | Best mode | Accuracy | Macro-F1 |
| --- | --- | --- | --- | --- |
| 1 (best) | MAX-Q=128, MAX-C=384, mean pooling | diff | 0.5320 | 0.4379 |
| 2 | MAX-Q=64, MAX-C=448 | diff | 0.5300 | 0.4168 |
| 3 | CLS pooling | concat | 0.5260 | 0.4061 |
| 4 | C=100 | diff | 0.5160 | 0.4219 |
| 5 | CTX-LIMIT=1 | concat | 0.4680 | 0.3541 |
| 6 | CTX-LIMIT=5 | diff | 0.5440 | 0.4325 |
| 7 | CTX-LIMIT=None | concat-diff | 0.5380 | 0.4332 |

### 1.4 Error Analysis

The PubMedQA expert-labelled dataset has a heavily skewed class distribution (yes: 55.2%, no: 33.8%, maybe: 11.0%). This imbalance means that models are incentivised to favour the majority class, and the small number of maybe instances means that even a few misclassifications dramatically reduce per-class recall.

Table 3 presents the comprehensive confusion matrix obtained by using the best model for all 10 folds.

Table 3:  Confusion Matrix

|  | Pred no | Pred yes | Pred maybe |
| --- | --- | --- | --- |
| Actual no | 78 | 69 | 22 |
| Actual yes | 75 | 175 | 26 |
| Actual maybe | 15 | 27 | 13 |

There are three main error patterns. 

First, it often confuses yes and no. This means it struggles to distinguish studies that support a hypothesis from those that reject it. This is likely because both types use similar biomedical language, while the key difference depends on interpreting statistical results, which frozen embeddings fail to capture.

Second, the model performs poorly on the maybe class. Out of 55 cases, only 13 are correctly predicted, while most are misclassified as yes or no. The maybe label requires understanding conditional or mixed conclusions, which the model cannot handle well, so it defaults to more certain categories.

Third, the no class has low recall. This is because no answers often include positive-sounding phrases like no significant difference, which the model may mistake for supportive findings.

### 1.5 Discussion

Replacing TF-IDF features with BiomedBERT embeddings gives a clear gain in balanced performance (macro-F1: 0.353 to 0.438), mainly by enabling the model to identify some maybe instances and improve no class recognition. The diff combination being the most effective supports the idea that the semantic distance between question and context is a useful signal for this task.

However, several limitations apply. BiomedBERT is frozen, meaning representations are not adapted to this specific classification task, and Logistic Regression can only draw linear decision boundaries in the embedding space. This limits the model’s ability to reason over quantitative content, which the PubMedQA paper identifies as the core difficulty of the task. Additionally, with only around 450 training instances per fold, the classifier is working in a low-data regime.

Despite these limitations, BiomedBERT with diff combination consistently outperforms the TF-IDF baseline across all metrics, demonstrating the value of domain-specific contextual representations even in a frozen, non-fine-tuned setting. These findings motivate the use of end-to-end fine-tuning in subsequent experiments.