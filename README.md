# LLE
Code implementation for paper [Lifelong Explainer for Lifelong Learners](https://aclanthology.org/2021.emnlp-main.233/) at EMNLP 2021, by Xuelin Situ, Sameen Maruf, Ingrid Zukerman, Cecile Paris and Reza Haffari.

# Requirements and Installation

- Python version >= 3.6.8
- PyTorch version >= 1.7.0
- HuggingFace transformers version >= 1.2.0
- [LIME](https://github.com/marcotcr/lime) >= 0.1.1.36
- [shap](https://github.com/slundberg/shap) == 0.29.3

# Experiments (steps to replicate the results from the paper)
1. **Train lifelong black-box classifier** >> *preprocess.train_lifelong_classifier.py*
2. **Collect explanations from different teachers** >> *preprocess.collect_teacher_explanations.py*

3. **Train LLE explainer (also refer to folder hyperparameters)** >> *lifelongexplanation.py*

4. **Faithfulness evaluation** >> *evaluation.compare_faithfulness.py*

5. **Stability evaluation** >> *evaluation.compare_stability.py*

6. **Efficiency evaluation** >> *evaluation.compare_efficiency.py*

7. **Ablation Study of Experience Replay**:
   - Faithfulness >> *evaluation.ablation_faithfulness.py*
   - Stability >> *evaluation.ablation_stability.py*