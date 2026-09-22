# Multi-Label ESG Classification with Transformer Models

**Fernando Nakamoto**

Transformer-based models for multi-label classification of ESG (Environmental,
Social, and Governance) text. Using datasets derived from
[ClimateBERT](https://arxiv.org/abs/2110.12010), the project models several
dimensions of sustainability-related disclosures — sentiment, specificity, and
TCFD category — and proposes a conceptual architecture for a scalable,
taxonomy-aligned ESG classifier built on large language models.

## Overview

Most ESG data is unstructured, embedded in reports, disclosures, and financial
documents, which creates a need for automated systems that can extract and
classify the relevant signals. Real-world sustainability disclosures are
inherently multi-dimensional: a single text segment may simultaneously express
multiple ESG attributes. This project explores that setting as a multi-label
text-classification problem and decomposes it into independent single-task
classifiers, one per ESG dimension.

## Data

The corpus is built from ClimateBERT datasets (`climate_detection`,
`climate_sentiment`, `climate_specificity`, `tcfd_recommendations`,
`environmental_claims`). Datasets that share a common subset of text instances
are merged via an inner join on the text field, producing aligned labels for
sentiment, specificity, and TCFD classification. Datasets without exact text
overlap are excluded from the unified corpus. The notebook downloads the data
at run time via the Hugging Face `datasets` library.

## Method

- Fine-tune pretrained transformer encoders for sequence classification.
- Decompose the multi-label problem into independent single-task classifiers.
- Stratified train/validation split, tokenization, training with the Hugging
  Face `Trainer`, and evaluation on validation and test sets.
- Analysis of training dynamics, confusion matrices, error cases, and joint
  (multi-label) performance.

## Results

| Task          | Accuracy | F1 (macro) |
|---------------|----------|------------|
| Specificity   | 0.81     | 0.79       |
| Sentiment     | 0.78     | 0.76       |
| TCFD          | 0.70     | 0.50       |

Joint (all labels correct simultaneously): accuracy 0.46, macro F1 0.245.
Performance decreases as task complexity increases — specificity (clear
linguistic signals) is easiest, TCFD (semantically overlapping categories) is
hardest. Errors accumulate across dimensions in the joint setting.

## How to run

1. Create an environment with Python 3.11 and install the dependencies:
   ```bash
   pip install torch transformers datasets scikit-learn pandas numpy matplotlib
   ```
2. Open `FernandoNakamoto_Notebook-v1.ipynb` and run the cells top to bottom.
   The notebook downloads the datasets and trains one classifier per ESG
   dimension. A GPU is recommended for training.

A rendered copy of the full run is available in
`FernandoNakamoto_Notebook-v1.pdf`.

> Model checkpoints, dataset caches, and media are excluded from version control
> (see `.gitignore`); they are regenerated when the notebook is run.

## References

[1] Webersinke, N., Kraus, M., Bingler, J., & Leippold, M. (2021).
*ClimateBERT: A Pretrained Language Model for Climate-Related Text.*
arXiv:2110.12010.

## License

Released under the [MIT License](LICENSE). © 2026 Fernando Nakamoto.
