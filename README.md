<style>
@font-face {
  font-family: "Berkeley Mono";
  src: url("./include/BerkeleyMono-Regular.woff2") format("woff2");
}
</style>

# Multimodal-LLM-Review
Some Overviews and Personal Readings on the Multimodal Large Model.

## Table of Content

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Multimodal-LLM-Review](#multimodal-llm-review)
  - [Table of Content](#table-of-content)
  - [Paper List](#paper-list)
    - [1 What is LLM-as-a-Judge?](#1-what-is-llm-as-a-judge)
    - [2 How to use LLM-as-a-Judge?](#2-how-to-use-llm-as-a-judge)
      - [2.1 In-Context Learning](#21-in-context-learning)
        - [Generating scores](#generating-scores)
        - [Solving Yes/No questions](#solving-yesno-questions)
        - [Conducting pairwise comparisons](#conducting-pairwise-comparisons)
        - [Making multiple-choice selections](#making-multiple-choice-selections)
      - [2.2 Model Selection](#22-model-selection)
        - [General LLM](#general-llm)
        - [Fine-tuned LLM](#fine-tuned-llm)
      - [2.3 Post-processing Method](#23-post-processing-method)
        - [Extracting specific tokens](#extracting-specific-tokens)
        - [Constrained decoding](#constrained-decoding)
        - [Normalizing the output logits](#normalizing-the-output-logits)
        - [Selecting sentences](#selecting-sentences)
      - [2.4 Evaluation Pipeline](#24-evaluation-pipeline)
        - [LLM-as-a-Judge for Models](#llm-as-a-judge-for-models)
        - [LLM-as-a-Judge for Data](#llm-as-a-judge-for-data)
        - [LLM-as-a-Judge for Agents](#llm-as-a-judge-for-agents)
        - [LLM-as-a-Judge for Reasoning/Thinking](#llm-as-a-judge-for-reasoningthinking)
    - [3 How to improve LLM-as-a-Judge?](#3-how-to-improve-llm-as-a-judge)
      - [3.1 Design Strategy of Evaluation Prompts](#31-design-strategy-of-evaluation-prompts)
        - [Few-shot promping](#few-shot-promping)
        - [Evaluation steps decomposition](#evaluation-steps-decomposition)
        - [Evaluation criteria decomposition](#evaluation-criteria-decomposition)
        - [Shuffling contents](#shuffling-contents)
        - [Conversion of evaluation tasks](#conversion-of-evaluation-tasks)
        - [Constraining outputs in structured formats](#constraining-outputs-in-structured-formats)
        - [Providing evaluations with explanations](#providing-evaluations-with-explanations)
      - [3.2 Improvement Strategy of LLMs' Abilities](#32-improvement-strategy-of-llms-abilities)
        - [Fine-tuning via Meta Evaluation Dataset](#fine-tuning-via-meta-evaluation-dataset)
        - [Iterative Optimization Based on Feedbacks](#iterative-optimization-based-on-feedbacks)
      - [3.3 Optimization Strategy of Final Results](#33-optimization-strategy-of-final-results)
        - [Summarize by multiple rounds](#summarize-by-multiple-rounds)
        - [Vote by multiple LLMs](#vote-by-multiple-llms)
        - [Score smoothing](#score-smoothing)
        - [Self validation](#self-validation)
    - [4 How to evaluate LLM-as-a-Judge？](#4-how-to-evaluate-llm-as-a-judge)
      - [4.1 Basic Metric](#41-basic-metric)
      - [4.2 Bias](#42-bias)
        - [Position Bias](#position-bias)
        - [Length Bias](#length-bias)
        - [Self-Enhancement Bias](#self-enhancement-bias)
        - [Other Bias](#other-bias)
      - [4.3 Adversarial Robustness](#43-adversarial-robustness)
    - [5 Application](#5-application)
      - [5.1 Machine Learning](#51-machine-learning)
        - [Text Generation](#text-generation)
        - [Reasoning](#reasoning)
        - [Retrieval](#retrieval)
      - [5.2 Social Intelligence](#52-social-intelligence)
      - [5.3 Multi-Modal](#53-multi-modal)
      - [5.4 Other Specific Domains](#54-other-specific-domains)
        - [Finance](#finance)
        - [Law](#law)
        - [AI for Science](#ai-for-science)
        - [Others](#others)
    - [6 Challenges](#6-challenges)
      - [6.1 Reliability](#61-reliability)
        - [Overconfidence](#overconfidence)
      - [6.2 Robustness](#62-robustness)

<!-- END doctoc -->


## Paper List

### 1 What is LLM-as-a-Judge?

- S. Tong et al., “Beyond Language Modeling: An Exploration of Multimodal Pretraining,” arXiv.org, 2026. https://arxiv.org/abs/2603.03276 (accessed May 28, 2026).
- C. Li et al., “Unifying Contrastive and Generative Objectives for Visual Understanding and Text-to-Image Generation,” arXiv.org, 2026. https://arxiv.org/abs/2603.02667 (accessed May 28, 2026).
- C. Huynh, M. Luong, and A. Shrivastava, “Efficient and High-Fidelity Omni Modality Retrieval,” arXiv.org, 2026. https://arxiv.org/abs/2603.02098 (accessed May 28, 2026).
- J. Huang et al., “Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy,” arXiv.org, 2026. https://arxiv.org/abs/2603.02123 (accessed May 28, 2026).
- N. Nagaraja, L. Zhang, Z. Wang, B. Zhang, and P. Patil, “Image-based Prompt Injection: Hijacking Multimodal LLMs through Visually Embedded Adversarial Instructions,” 2025 3rd International Conference on Foundation and Large Language Models (FLLM), pp. 916–922, Nov. 2025, doi: https://doi.org/10.1109/fllm67465.2025.11391218.
- Y. Sun, K. Li, P. Guo, J. Liu, and Q. Tan, “Mario: Multimodal Graph Reasoning with Large Language Models,” arXiv.org, 2026. https://arxiv.org/abs/2603.05181 (accessed May 28, 2026).
- J. Fan and W. Song, “VisionPangu: A Compact and Fine-Grained Multimodal Assistant with 1.7B Parameters,” arXiv.org, 2026. https://arxiv.org/abs/2603.04957 (accessed May 28, 2026).
- Ngong, Ivoline C, Z. Reza, and J. P. Near, “Differentially Private Multimodal In-Context Learning,” arXiv.org, 2026. https://arxiv.org/abs/2603.04894 (accessed May 28, 2026).
- L. Hu et al., “MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models,” arXiv.org, 2026. https://arxiv.org/abs/2603.04800 (accessed May 28, 2026).
- Y. Shi, Q. Zhao, T. Jiang, X. Zeng, Y. Wang, and L. Wang, “RIVER: A Real-Time Interaction Benchmark for Video LLMs,” arXiv.org, 2026. https://arxiv.org/abs/2603.03985 (accessed May 28, 2026).