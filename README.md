# SAINT: Semantic Alignment with Inferential Transformers

This repository contains research code for SAINT: (S)semantic (A)alignment with (I)(N)ferential (T)transformers. SAINT is a new evaluation mechanism and a learning strategy for de-biased and interpretable NLI models. 

## SAINT as an Evaluation Mechanism
We propose an evaluation method beyond the traditional accuracy/F1-based evaluation for NLI models. In particular, SAINT co-evaluates a model's ability to:
1. predict the correct NLI label 
2. generate evidence (**semantically alignment relations**) supporting its prediction. 

This evaluation has two formats: 
1. Multi-task style train a transformer to do multi-task learning on nli and semantic alignment. During the evaluation, the model first predicts a label then generates a list of evidence based on this label. An example is considered correct only when the label is correct, and the evidence mostly matches the gold targets.
2. Single-task style trains a transformer to predict the label and list the evidence from a single input prompt. An example is considered correct only when the label is correct, and the evidence mostly matches the gold targets.

## SAINT as a Learning Strategy
We demonstrate that co-learning sentence classification and evidence retrieval can encourage the model to apply the correct reasoning and information to arrive at the final label. Such learning can help reduce the model's dependency on superficial cues and annotation artifacts existing in datasets.
