# Q3-report

**the original question: Is it possible to predict answer quality from characteristics of the question? Consider whether regression or a small neural network can decide if a question is likely to produce a good quality answer.**

1. After all the experiments were completed, the results of each group were saved as a CSV file. The file contains four columns: the encoded PMID that uniquely identifies the specific question, the original data’s classification label y_true, the predicted label y_pred, and the “correct” indicator indicating whether the prediction was accurate. In question 3, we will explore whether the quality of the answer can be predicted from the characteristics of the medical question. This requires us to extract key features from the question, such as the length of the question, whether it requires reasoning, and the number of verbs in the question.
2. Due to the varying predictive capabilities of different QA models, if we directly use the correct labels of a single model as the learning target, the performance of the quality prediction model will be affected by the bias of that specific QA model. To remove this influence, we averaged the correct labels from four groups of experiments to obtain a continuous difficulty score ranging from 0 to 1. This score reflects the inherent difficulty of the problem rather than the performance deviation of a specific model. Thus, we transformed the task from a classification problem to a regression problem and used a regression model to predict this difficulty score.
3. Regarding this question, the evaluation metrics we chose are MSE and Pearson r, which respectively measure the absolute gap between the predicted values and the actual values, as well as whether the trends of the predicted values and the actual values are consistent. We selected four regression models for the model: Ridge, Lasso, BayesianRidge, and MLP. As shown in the table, the Lasso model ultimately performed the best in both MSE and Pearson r evaluation metrics, because it can automatically select useful features, the model has strong interpretability, and the L1 penalty can effectively prevent overfitting in scenarios with high dimensions and small samples. However, MLP performed relatively weakest. This might be because in a small data scale of 500 samples, nonlinear models are prone to overfitting, while linear models are more robust.

|  | MSE | Pearson r |
| --- | --- | --- |
| Ridge | 0.080 ± 0.010 | 0.581 ± 0.058 |
| Lasso | 0.076 ± 0.009 | 0.603 ± 0.056 |
| BayesianRidge | 0.079 ± 0.010 | 0.584 ± 0.055 |
| MLP | 0.088 ± 0.010 | 0.533 ± 0.044 |
4. As shown in the table, this table illustrates the five problem characteristics that have the greatest impact on the quality of answers under the Lasso model. They respectively represent questions that can be answered without reasoning (simple questions), questions that require multiple steps of reasoning (complex questions), cases where the above two labels are inconsistent, the presence of population-related words in the question, and the number of verbs in the question. The first three “reasoning” are the built-in annotations in the PubMedQA dataset and are the strongest signals for predicting difficulty. This indicates that whether a question requires multiple-step reasoning is the strongest indicator for predicting the performance of the question answering system, far more influential than question length, entity quantity, or vocabulary complexity. Overall, simple linear models leveraging textual features and metadata are sufficient to achieve moderate predictive performance on this task.

| Feature | Coefficient |
| --- | --- |
| **reasoning_free** | 0.125940 |
| **reasoning_required** | 0.089405 |
| **reasoning_mismatch** | 0.063279 |
| **q_has_population** | 0.012606 |
| **q_num_verbs** | 0.005520 |