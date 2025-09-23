# Report Outline for HA6: Sentiment Analysis with BERT

## 1. Introduction

**Background**: Pre-trained language models like BERT have shown remarkable performance on downstream tasks such as classification. Sentiment analysis of the IMDB dataset is a standard benchmark to test model transferability.

**Motivation**: While fine-tuning the whole BERT is powerful, it is computationally expensive. Moreover, different layers of BERT encode different linguistic information. A key question is: How transferable and separable are the representations from different BERT layers for sentiment classification?

**Research Questions**:
- Is linear probing (training only a classifier on frozen features) sufficient for sentiment classification?
- What is the effect of using middle-layer vs last-layer representations?
- Can partial fine-tuning (unfreezing a few layers) achieve a good trade-off between accuracy and efficiency?
- How robust are these strategies under low-resource settings?

**Contributions**: This report systematically investigates layer-wise probing, partial fine-tuning, and resource-sensitive scenarios for the IMDB sentiment task.

## 2. Methods

**Dataset**: IMDB Dataset (50k samples, labeled positive/negative). Randomly shuffled, split into training and test sets.

**Model**: Pre-trained BERT base model with a classification head.

**Experimental Factors**:
- **Representation Layer**: last layer CLS, intermediate layers, multi-layer pooling.
- **Training Strategy**: linear probing, partial fine-tuning (last 2 / last 4 layers), full fine-tuning.
- **Data Scale**: full dataset vs low-resource (5k training samples).
- **Optimization Setting**: baseline hyperparameters, plus learning rate variation.
- **Evaluation Metrics**: Test accuracy (primary), F1-score (secondary), training time, number of trainable parameters.

## 3. Experiments

**Baseline Setup**: Full fine-tuning, last-layer CLS token representation, full training set.

**Experiment Grid (≥10 runs)**:
- Linear probing vs full fine-tuning.
- Intermediate-layer probing (e.g., 8th layer).
- Multi-layer pooling representation.
- Partial fine-tuning (last 2 / last 4 layers).
- Low-resource setting (5000 samples).
- Learning rate variations (e.g., 3e-5 vs 1e-5).

**Implementation Details**: Batch size, max epochs, optimizer, dropout; state any consistent defaults across runs.

## 4. Results

**Table of Results**: Each row = one experiment, report accuracy, F1, training time, #parameters updated.

**Visualization**: Optional bar chart / line plot comparing strategies (e.g., probe vs fine-tune).

**Key Comparisons**:
- Linear probing vs fine-tuning.
- Intermediate vs last layers.
- Full vs partial fine-tuning.
- Full-data vs low-resource scenarios.

## 5. Discussion

**Performance Patterns**: Which strategies achieved the highest accuracy? Which were most efficient?

**Layer-wise Insights**: Did intermediate layers provide useful sentiment features compared to the last layer?

**Efficiency Trade-offs**: How well did partial fine-tuning balance accuracy and computational cost?

**Robustness Under Low Resource**: Which methods worked better with only ~5k training samples?

**Hyperparameter Sensitivity**: How stable was fine-tuning across learning rates?

**Limitations**: Resource constraints, limited hyperparameter grid, scope of dataset.

## 6. Conclusion

**Summary of Findings**: (e.g., linear probing reveals separability of BERT features but full fine-tuning yields the best accuracy; partial fine-tuning significantly reduces cost with little performance drop).

**Broader Implications**: Insights can generalize to other NLP classification tasks, especially in low-resource and efficiency-critical scenarios.

**Future Work**: Extend to multi-class sentiment, cross-lingual transfer, or probing tasks beyond sentiment classification.