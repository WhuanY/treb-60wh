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

In this section, we present a series of experiments on the IMDB sentiment analysis task to systematically evaluate the effectiveness of different fine-tuning strategies applied to BERT. As a baseline, we consider full fine-tuning of the model using the last-layer [CLS] representation on the entire training set. Beyond the baseline, we construct a grid of experiments including linear probing, where all BERT parameters are frozen and only a classification head is trained; probing of intermediate representations such as the 8th layer; multi-layer pooling (e.g., averaging representations from both the 8th and 12th layers); and partial fine-tuning, where only the top two or four layers are unfrozen during training. To assess robustness under limited supervision, additional experiments are performed on reduced training data consisting of 5,000 samples. All models are optimized with AdamW using a fixed learning rate of 2e-5, no warmup steps, and a batch size of 48. Each run is trained for at most 10 epochs, with early stopping applied if the validation performance does not improve for two consecutive evaluations, in which case the best-performing checkpoint on the validation set is retained for testing. This unified configuration ensures that results across different strategies are directly comparable, highlighting the relative trade-offs between accuracy, efficiency, and representational transferability.


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