# EX1-report

# 4. Experimental Evaluation

### 4.1 Performance Metrics and Their Limitations

We evaluate all models using two metrics: accuracy and macro-averaged F1 score. Accuracy measures the proportion of correctly classified instances across all three labels. However, the PubMedQA expert-labelled dataset has a heavily skewed class distribution (yes: 55.2%, no: 33.8%, maybe: 11.0%), which makes accuracy misleading on its own. A classifier that always predicts "yes" achieves 55.2% accuracy while doing no useful work. We therefore treat macro-F1 as our primary metric, as it averages F1 equally across all three classes regardless of frequency, and penalises models that fail to recognise the minority maybe class. Per-class F1 scores for no, yes and maybe are also reported to reveal class-specific failure modes.

A limitation of accuracy is that it can increase simply by becoming more confident on the majority class, hiding deterioration on maybe. A limitation of macro-F1 is that it treats failure on maybe (11% of data) as equally important as failure on yes (55%), which may not reflect real clinical priorities. For the purpose of evaluating whether models can genuinely distinguish all three answer categories, macro-F1 is the more appropriate choice.

### 4.2 Experimental Setup

All experiments use the expert-labelled subset of PubMedQA (PQA-L), which contains 1,000 instances. We follow the standard evaluation protocol: 10-fold cross-validation on a fixed 500-instance split, with the remaining 500 instances reserved as a held-out test set. 

In Experiment 1, the frozen BiomedBERT model was used to encode the question and context separately. This model was pre-trained on PubMed abstract and full-text articles. The questions were limited to 128 tokens and the contexts to 384 tokens. Then, the output of all tokens was averaged pooled to obtain a 768-dimensional vector for each input. Next, we test four vector combination methods: concat, diff, prod, and concat-diff. Before training, we standardise all features. Subsequently, we use a logistic regression classifier with the same settings as the baseline to train these vectors.
To save time, we calculated all the BiomedBERT embeddings for the folds at once before the classification loop, thus avoiding redundant encoding. We tested a total of seven parameter configurations, including token length, pooling method, regularisation strength C, and context length limit. The configuration that performed the best in the final report was selected as the main result.

All the experiments were completed on a NVIDIA T4 GPU. The encoding for 10 folds took approximately 130 seconds in total. The training time for each fold of logistic regression was less than 0.5 seconds.

### 4.3 Results

Table 1: 10-Fold Cross-Validation Results

| Model | Combination | Accuracy | Macro-F1 | F1 no | F1 yes | F1 maybe |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + LR (Baseline) | n/a | 0.5600 | 0.3439 | 0.3413 | 0.6904 | 0.0000 |
| BiomedBERT + LR | concat | 0.5280 | 0.4195 | 0.4613 | 0.6418 | 0.1554 |
| BiomedBERT + LR | diff | 0.5320 | 0.4379 | 0.4624 | 0.6369 | 0.2143 |
| BiomedBERT + LR | prod | 0.4960 | 0.3831 | 0.3933 | 0.6229 | 0.1329 |
| BiomedBERT + LR | concat-diff | 0.5280 | 0.4228 | 0.4725 | 0.6398 | 0.1561 |

In Experiment 1, the best result was achieved using diff combination method, with macro-F1 score of 0.4379, which was 27.33% higher than the baseline. The overall accuracy rate has slightly decreased, from 0.560 to 0.532.This happens because the model now gives some probability to the maybe class instead of always choosing the majority class. The F1 score for the “maybe” category in the baseline model was 0.000, indicating that it never predicted this category. In contrast, using diff along with BiomedBERT and logistic regression increased the F1 score for the “maybe” category to 0.2143.

Table 2 presents the result of the parameter sensitivity analysis. Among the seven settings, the diff combination achieved the best or nearly best macro-F1 score in five cases.
When we reduced the context to a single paragraph (CTX-LIMIT = 1), the performance dropped most significantly, with the macro-F1 score falling to 0.354. This indicates that the model requires information from multiple paragraphs.
Mean pooling outperforms CLS pooling, with scores of 0.4379 and 0.4061 respectively. When we increase C from 10 to 100, the performance slightly declines, indicating that an appropriate level of regularization is more effective.

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

### 4.4 Error Analysis

Table 3 presents the comprehensive confusion matrix obtained by using the best model for all 10 folds.

Table 3:  Confusion Matrix

|  | Pred no | Pred yes | Pred maybe |
| --- | --- | --- | --- |
| Actual no | 78 | 69 | 22 |
| Actual yes | 75 | 175 | 26 |
| Actual maybe | 15 | 27 | 13 |

It is clearly evident that there are three error patterns. Firstly, this model confuses “yes” and “no” in both positive and negative directions. It wrongly classifies 69 “no” cases as “yes”, and wrongly classifies 75 “yes” cases as “no”. This shows that the model has difficulty telling apart studies that support a hypothesis from those that reject it. This is expected, as both classes often contain similar biomedical vocabulary and the key difference lies in the direction of statistical findings, which frozen embeddings cannot reliably capture.

Second, maybe is still mostly misclassified. Of 55 actual maybe instances across all folds, only 13 are correctly identified, while 27 are predicted as yes and 15 as no. The maybe label is used when a paper discusses conditions under which both true and false interpretations apply, which requires understanding conditional reasoning in the conclusion. Without this, models default to the nearest confident class.

Third, the no class has the lowest recall (78 out of 169, recall = 0.46). No answers often contain positive-sounding language such as “there was no significant difference”, which embeddings may not reliably separate from affirmative findings.

### 4.5 Discussion

Replacing TF-IDF features with BiomedBERT embeddings gives a clear gain in balanced performance (macro-F1: 0.344 to 0.438), mainly by enabling the model to identify some maybe instances and improve no class recognition. The diff combination being the most effective supports the idea that the semantic distance between question and context is a useful signal for this task.

However, several limitations apply. BiomedBERT is frozen, meaning representations are not adapted to this specific classification task, and Logistic Regression can only draw linear decision boundaries in the embedding space. This limits the model’s ability to reason over quantitative content, which the PubMedQA paper identifies as the core difficulty of the task. Additionally, with only around 450 training instances per fold, the classifier is working in a low-data regime.

Compared to the baseline, the decrease in accuracy does not indicate a deterioration in performance. The reason why the baseline has a higher accuracy rate is that it mainly predicts the majority category “yes” and never predicts “maybe”.
The BiomedBERT model provides more balanced predictions across various categories. Although the accuracy has slightly decreased, this trade-off is more meaningful in the current evaluation context. These findings motivate the use of end-to-end fine-tuning in subsequent experiments.