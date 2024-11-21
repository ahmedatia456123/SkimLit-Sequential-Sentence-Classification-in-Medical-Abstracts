# SkimLit: Sequential Sentence Classification in Medical Abstracts

## Objective

In the previous notebook, **NLP Fundamentals in TensorFlow**, we covered essential natural language processing (NLP) concepts, including **tokenization** (converting words into numerical representations) and **embedding creation** (transforming words into vector representations). In this project, we will apply these techniques to real-world data.

Specifically, we aim to replicate the deep learning model described in the 2017 paper, *PubMed 200k RCT: A Dataset for Sequential Sentence Classification in Medical Abstracts*.

## Dataset Overview

When this paper was released, it introduced the **PubMed 200k RCT dataset**, which consists of **~200,000 labeled Randomized Controlled Trial (RCT) abstracts**. The purpose of this dataset was to explore how NLP models can classify sentences that appear in sequential order within an abstract.

The task we aim to solve is: **Given the abstract of an RCT, what role does each sentence serve in the abstract?** This involves classifying sentences into different categories such as **Objective**, **Methods**, **Results**, and **Conclusions**.

## Problem and Solution

### Problem:
The number of RCT papers continues to grow, and many of these papers lack structured abstracts. This makes it harder for researchers to quickly understand the content, slowing down literature review processes.

### Solution:
We will develop an NLP model to classify sentences within abstracts into roles like **Objective**, **Methods**, **Results**, and **Conclusions**. This will allow researchers to skim through abstracts more efficiently (hence **SkimLit** 🤓🔥) and dive deeper into sections when needed.

## Model Overview

Our model will take an unstructured abstract and predict the **section label** for each sentence. Below is an example of the **input** and **output** for our model.

### Example Input:

A medical abstract from PubMed (numerical symbols replaced with `@` for simplicity):

To investigate the efficacy of @ weeks of daily low-dose oral prednisolone in improving pain, mobility, and systemic low-grade inflammation in the short term and whether the effect would be sustained at @ weeks in older adults with moderate to severe knee osteoarthritis (OA). A total of @ patients with primary knee OA were randomized @:@; @ received @ mg/day of prednisolone and @ received placebo for @ weeks. Outcome measures included pain reduction and improvement in function scores and systemic inflammation markers. Pain was assessed using the visual analog pain scale (@-@ mm). Secondary outcome measures included the Western Ontario and McMaster Universities Osteoarthritis Index scores, patient global assessment (PGA) of the severity of knee OA, and @-min walk distance (@MWD). Serum levels of interleukin @ (IL-@), IL-@, tumor necrosis factor (TNF)-, and high-sensitivity C-reactive protein (hsCRP) were measured. There was a clinically relevant reduction in the intervention group compared to the placebo group for knee pain, physical function, PGA, and @MWD at @ weeks. The mean difference between treatment arms (@% CI) was @(@-@ @), p<@; @(@-@ @), p<@; @(@-@ @), p<@; and @(@-@ @), p<@, respectively. Further, there was a clinically relevant reduction in the serum levels of IL-@, IL-@, TNF-, and hsCRP at @ weeks in the intervention group when compared to the placebo group. These differences remained significant at @ weeks. The Outcome Measures in Rheumatology Clinical Trials-Osteoarthritis Research Society International responder rate was @% in the intervention group and @% in the placebo group (p<@). Low-dose oral prednisolone had both a short-term and a longer sustained effect resulting in less knee pain, better physical function, and attenuation of systemic inflammation in older patients with knee OA (ClinicalTrials.gov identifier NCT@).

perl
Copy code

### Example Output:

The model will classify each sentence and return a structured output:

['###24293578\n', 'OBJECTIVE\tTo investigate the efficacy of @ weeks of daily low-dose oral prednisolone in improving pain, mobility, and systemic low-grade inflammation in the short term and whether the effect would be sustained at @ weeks in older adults with moderate to severe knee osteoarthritis (OA).\n', 'METHODS\tA total of @ patients with primary knee OA were randomized @:@; @ received @ mg/day of prednisolone and @ received placebo for @ weeks.\n', 'METHODS\tOutcome measures included pain reduction and improvement in function scores and systemic inflammation markers.\n', 'METHODS\tPain was assessed using the visual analog pain scale (@-@ mm).\n', 'METHODS\tSecondary outcome measures included the Western Ontario and McMaster Universities Osteoarthritis Index scores, patient global assessment (PGA) of the severity of knee OA, and @-min walk distance (@MWD).\n', 'METHODS\tSerum levels of interleukin @ (IL-@), IL-@, tumor necrosis factor (TNF)-, and high-sensitivity C-reactive protein (hsCRP) were measured.\n', 'RESULTS\tThere was a clinically relevant reduction in the intervention group compared to the placebo group for knee pain, physical function, PGA, and @MWD at @ weeks.\n', 'RESULTS\tThe mean difference between treatment arms (@% CI) was @(@-@ @), p<@; @(@-@ @), p<@; @(@-@ @), p<@; and @(@-@ @), p<@, respectively.\n', 'RESULTS\tFurther, there was a clinically relevant reduction in the serum levels of IL-@, IL-@, TNF-, and hsCRP at @ weeks in the intervention group when compared to the placebo group.\n', 'RESULTS\tThese differences remained significant at @ weeks.\n', 'RESULTS\tThe Outcome Measures in Rheumatology Clinical Trials-Osteoarthritis Research Society International responder rate was @% in the intervention group and @% in the placebo group (p<@).\n', 'CONCLUSIONS\tLow-dose oral prednisolone had both a short-term and a longer sustained effect resulting in less knee pain, better physical function, and attenuation of systemic inflammation in older patients with knee OA (ClinicalTrials.gov identifier NCT@).\n', '\n']

markdown
Copy code

## Problem Statement

The increasing number of RCT papers without structured abstracts makes it difficult for researchers to quickly navigate through the literature.

## Proposed Solution

Create an NLP model that classifies sentences in abstracts according to their roles (e.g., Objective, Methods, Results, etc.), enabling researchers to **skim** the literature efficiently and dive deeper into sections when necessary.

## Data Source

The data for this project comes from the **PubMed 200k RCT dataset**, which is designed specifically for **sequential sentence classification** in medical abstracts.

The model architecture we aim to replicate is based on the paper: **Neural networks for joint sentence classification in medical paper abstracts**.

## 📖 Resources

Before diving into the code, it's recommended to get familiar with the following papers to understand the underlying concepts:

- [PubMed 200k RCT: A Dataset for Sequential Sentence Classification in Medical Abstracts](https://arxiv.org/abs/1612.05251)
- [Neural Networks for Joint Sentence Classification in Medical Paper Abstracts](https://arxiv.org/abs/1612.05251)

## What We’ll Cover

We’re about to take everything we've learned from the **NLP Fundamentals** notebook and build one of our largest NLP models to date. Here's an outline of the steps:

1. **Download the PubMed RCT200k dataset** from GitHub.
2. **Preprocess the data** to prepare it for modeling.
3. **Set up a series of experiments** to test different model architectures.
4. **Create a baseline** using a TF-IDF classifier.
5. **Build deep models** incorporating different types of embeddings:
   - Token embeddings
   - Character embeddings
   - Pretrained embeddings
   - Positional embeddings
6. **Construct our first multimodal model** that takes multiple types of data inputs.
7. **Replicate the model architecture** from the paper.
8. **Identify the most incorrect predictions** and improve our model.
9. **Make predictions** on PubMed abstracts from real-world data.

