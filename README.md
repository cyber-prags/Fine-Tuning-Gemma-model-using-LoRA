# Fine-Tuning Gemma Models in Keras using LoRA
![image](https://github.com/user-attachments/assets/5537f6d2-d771-413a-944f-a660b5aad4df)


This project demonstrates how to fine-tune Google's Gemma language models using **Keras** and **Low-Rank Adaptation (LoRA)**. The notebook walks through model loading, lightweight fine-tuning with LoRA, and inference before and after fine-tuning. This guide is perfect for machine learning practitioners who want to fine-tune LLMs efficiently using limited resources.

---

## 🎓 Table of Contents
- [Project Overview](#project-overview)
- [What is Gemma?](#what-is-gemma)
- [What is LoRA?](#what-is-lora)
- [Environment and Dependencies](#environment-and-dependencies)
- [Dataset Used](#dataset-used)
- [Model Inference (Zero-shot)](#model-inference-zero-shot)
- [Fine-Tuning with LoRA](#fine-tuning-with-lora)
- [Post-Tuning Inference](#post-tuning-inference)
- [Key Techniques Explained](#key-techniques-explained)
- [Further Resources](#further-resources)

---

## 🌐 Project Overview

This notebook explores:
- Running zero-shot inference on a pre-trained Gemma model.
- Fine-tuning it using **LoRA** to make it more efficient.
- Evaluating the model's performance after fine-tuning.

The goal is to achieve **parameter-efficient fine-tuning** of large models.

---

## 🚀 What is Gemma?

**Gemma** is a family of lightweight open-weight language models developed by Google. They are designed to be:
- Easy to use
- Resource-efficient
- Instruction-tunable for downstream tasks

Gemma models are similar to LLaMA, Mistral, and other transformer-based autoregressive models, and are optimized for both research and production.

🔗 More on Gemma: [https://ai.google.dev/gemma](https://ai.google.dev/gemma)

---

## 🦖 What is LoRA?
![image](https://github.com/user-attachments/assets/6eafffca-9ff5-40b1-ac87-5deb0beb66db)


**LoRA (Low-Rank Adaptation)** is a fine-tuning technique that:
- Introduces small trainable matrices into the transformer architecture.
- Freezes the original pre-trained weights.
- Fine-tunes only the additional parameters, significantly reducing training time and memory.

### ✅ Benefits:
- Efficient training (memory and compute).
- Ideal for large language models.
- Helps avoid catastrophic forgetting.

📄 Paper: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

---

## 🧪 Environment and Dependencies

- `tensorflow` / `keras`
- `keras-nlp`
- `keras-core`
- `transformers`
- `huggingface_hub`
- Dataset: `databricks-dolly-15k` (see below)

---

## 📂 Dataset Used

### Databricks Dolly 15K
A collection of 15,000 instruction-response pairs, manually created by over 5,000 Databricks employees.

**Features:**
- Categories: Open QA, Brainstorming, Summarization, etc.
- Fully open-source and commercially usable.

📦 More info: [https://huggingface.co/datasets/databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

---

## 🤖 Model Inference (Zero-shot)

Before fine-tuning, the model is tested in its zero-shot form to:
- Generate answers to prompts.
- Demonstrate baseline performance.

**Technique Used:**  
**Top-K Sampling** (`k=5`) for diverse response generation.

```python
prompt = template.format(instruction="Explain photosynthesis in a way a child could understand.", response="")
sampler = keras_nlp.samplers.TopKSampler(k=5)
gemma_lm.compile(sampler=sampler)
gemma_lm.generate(prompt)
```

## 🛠 Fine-Tuning with LoRA
### Enabling LoRA
```
gemma_lm.backbone.enable_lora(rank=4)
```
Only 1.3M parameters are trainable (vs 2.5B originally)

Dramatically reduces compute requirement

### Optimizer and Loss Setup
```
optimizer = keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```
### Training
```
gemma_lm.fit(data, epochs=1, batch_size=1)
```
Why 1 Epoch and Batch Size 1?
Epoch = 1: For testing and quick experimentation

Batch Size = 1: Low memory use; good for debugging and limited-resource environments

## 🧠 Post-Tuning Inference
After training, inference is re-run on the same prompt:

Expected to be more aligned to specific instruction types

Shows improved understanding and alignment to the fine-tuned domain

## 🧬 Key Techniques Explained
### 🔩 LoRA (Low-Rank Adaptation)
Adds trainable low-rank matrices (A, B) to frozen weights

Fine-tunes efficiently with very few trainable parameters

### ⚖️ Top-K Sampling
Filters vocabulary to top k tokens based on probability

Balances creativity and relevance in text generation

### 🔁 AdamW Optimizer
Adam with decoupled weight decay

Helps reduce overfitting in Transformer models

### ⚖️ Sparse Categorical Crossentropy
Works with integer labels for classification

Efficient for language modeling tasks

### 📊 Further Resources
Gemma Models Documentation -> https://ai.google.dev/gemma

LoRA: Low-Rank Adaptation Paper -> https://arxiv.org/abs/2106.09685

Databricks Dolly 15K Dataset -> https://huggingface.co/datasets/databricks/databricks-dolly-15k

Top-K and Top-P Sampling Guide -> https://huggingface.co/blog/how-to-generate


