# introduction

Question answering systems (QAS) are a central goal of Natural Language Processing, aiming to provide precise answers to user input queries in natural language. Such systems can enable efficient access to large-scale information, significantly reducing the time and effort required compared to a manual search. This is particularly relevant in rapidly expanding, highly specialised domains such as biomedical research, where accessing accurate and relevant information can directly influence critical decisions made by clinicians and researchers. As a result, domain-specific QAS provide a useful tool to support evidence-based decision making by facilitating faster and more reliable information retrieval.
In our project, we aim to design and implement a QAS capable of answering biomedical questions using the PubMedQA dataset. Our primary objective is to investigate how different text representations and model architectures affect the quality of answers. By systematically comparing alternative approaches, we aim to identify which methods are most effective for handling the challenges of biomedical question answering.
In order to evaluate variations of our system we use PubMedQA; a dataset constructed from biomedical research papers. Each instance in the dataset consists of a question derived from a research title, a corresponding abstract (excluding the conclusion) that provides context, a long answer (the abstract conclusion), and a yes/no/maybe short answer. This structure allows the model architecture to be constructed either as classification or as answer generation, allowing for easy comparison of different AI methods. Furthermore, the inputs are provided as paired question-context instances allowing the task to focus on comprehension and reasoning of the text, instead of retrieval, enabling easier evaluation of model performance.

![Fig.1 Label Distribution in PQA-L](image.png)

Fig.1 Label Distribution in PQA-L

![Fig.2 Context Length Distribution in PQA-L](image%201.png)

Fig.2 Context Length Distribution in PQA-L

As shown in Fig.1, the dataset shows label imbalance, yes accounts for 55.2%, no for 33.8%, and maybe for only 11.0%. This distribution means that accuracy alone is an insufficient evaluation matric, and using macro-averaged F1 is needed. Besides, in Fig.2, contexts average 200 words in length, with some exceeding 400 words, indicating the challenge of extracting relevant information from dense biomedical contexts.

However, biomedical data can provide significant challenges. The language in research papers often contains complex terminology, domain-specific phrases and abbreviations which can be difficult for models to interpret. Additionally, abstracts are usually dense and information-rich, requiring models to identify and extract relevant details from long passages of text. Many questions also require inferential reasoning rather than simple keyword matching, further increasing difficulty. These challenges highlight the importance of selecting appropriate text representations and architecture in order to achieve high accuracy and reliability.

To address these challenges, we structure our experiment around two key axes. The first axis is text representation: we evaluate how different encoding strategies, from sparse statistical features to dense contextual embeddings, affect model performance. The second is the answer generation axis, where we compare different ways of predicting the final answer, from traditional classifiers to large language generative models. Combining these two axes, we systematically vary key components within a shared pipeline to understand their impact on performance.

~~A key factor influencing QAS performance is text representation. Traditional bag-of-words and TF-IDF representations are simple and easy to interpret but often fail to capture semantic relationships and contextual meaning. In contrast, more advanced embedding-based-methods can represent text in a way that better captures meaning and context which is particularly important in specialised domains. A further key design consideration is model selection. Classification approaches are typically more efficient, and performance is easier to evaluate, especially when answers are limited to predetermined labels. However, they may struggle to capture nuance. In comparison, generative models are more flexible and can produce stronger answer but provide additional challenges such as increased complexity and potential for error. Understanding how text representation and model architecture interact to affect answer quality is key to optimising performance.~~

A good solution will be able to accurately interpret both the question and context, understand domain-specific semantics, and produce answers that are reliable and consistent. First, it must handle the problem of class imbalance. The dataset’s label distribution is heavily imbalance among yes, no, and maybe, with maybe appearing far less frequently than yes and no. This means that even though a model achieves high accuracy, it may still fail systematically on the minority class. Thus, macro-averaged F1 is a more meaningful evaluation metric than accuracy alone. Second, models must reason over long, information-dense contexts, which is a particular challenge for sparse representations. Third, we focus not only on model performance, but also on computational costs, making the efficiency of each approach a relevant factor. Based on the research reported by Jin et al. 2019, a majority-class baseline achieves 55.2% accuracy, while reasoning-required human performance achieves 78.0%, providing a meaningful reference for our experiments.

Figure: Taken from Jin et al. 2019. An instance (Sakamoto et al. 2011) from PubMedQA dataset.

<aside>
💡

**Question:** Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?

**Context:**

(Objective) Recent studies have demonstrated that statins have pleiotropic effects, including anti-inflammatory effects and atrial fibrillation (AF) preventive effects [...]

(Methods) 221 patients underwent CABG in our hospital from 2004 to 2007. 14 patients with preoperative AF and 4 patients with concomitant valve surgery [...]

(Results) The overall incidence of postoperative AF was 26%. Postoperative AF was significantly lower in the Statin group compared with the Non-statin group (16% versus 33%, p=0.005). Multivariate analysis demonstrated that independent predictors of AF [...]

**Long Answer:**

(Conclusion) Our study indicated that preoperative statin therapy seems to reduce AF development after CABG. 

**Answer:** yes

</aside>

Jin, Q., Dhingra, B., Liu, Z., Cohen, W. and Lu, X., 2019, November. Pubmedqa: A dataset for biomedical research question answering. In *Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP)* (pp. 2567-2577).

Sakamoto, H., Watanabe, Y. and Satou, M., 2011. Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?. *Annals of thoracic and cardiovascular surgery*, *17*(4), pp.376-382.