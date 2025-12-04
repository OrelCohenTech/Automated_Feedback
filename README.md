# Beyond Grading: Automated Answer Correctness Evaluation 

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![HuggingFace](https://img.shields.io/badge/Transformers-4.30-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

> **Course Project:** Deep Learning & NLP | **Focus:** Educational Data Mining, Semantic Textual Similarity (STS), & Synthetic Data Generation.

---

##  Overview

**Beyond Grading** is an NLP framework designed to automate the evaluation of open-ended student answers in STEM domains (Computer Science, Algorithms, Data Structures). Unlike traditional Keyword Matching or simple binary classification, our system leverages **Deep Learning (Transformer-based Cross-Encoders)** and **GenAI Augmentation** to assess the *semantic correctness* of a student's response.

The goal is to solve the "Scalability vs. Quality" dilemma in modern education by providing automated, accurate, and consistent grading scores (0-1) that reflect deep understanding.

##  Key Features & Novelty

### 1. Cross-Encoder Architecture (Triple Context)
We move beyond simple cosine similarity. Our model processes three inputs simultaneously to capture nuance:
* **Context:** The Exam Question.
* **Ground Truth:** The Expert/Reference Answer.
* **Target:** The Student's Response.

### 2. Adversarial Synthetic Data Strategy 
Addressing the lack of labeled educational data and class imbalance, we developed a **GenAI Data Pipeline**:
* **Hard Negatives:** We use **GPT-4o** to deliberately generate "plausible but wrong" answers (responses that use correct terminology but contain logical flaws).
* **Implicit Labeling:** Labels are assigned automatically based on the generation prompt (e.g., specific misconception prompts = low score), removing the need for manual annotation.

### 3. Fine-Tuned Regression Head
The model is fine-tuned on a hybrid dataset (Real + Synthetic) to predict a continuous correctness score (0.0 - 1.0) using MSE Loss, optimized for educational nuances.

---

##  Tech Stack & Pipeline



[Image of machine learning pipeline diagram]

*(Place your pipeline image here in the assets folder)*

* **Language:** Python
* **DL Framework:** PyTorch, Hugging Face Transformers
* **Backbone Models:** `microsoft/deberta-v3-base`, `bert-base-uncased`
* **Data Generation:** OpenAI API (GPT-4o / Llama-3)
* **Experiment Tracking:** WandB (Weights & Biases) / TensorBoard

---

##  Dataset Specification

We utilize a hybrid data strategy:

| Dataset Source | Description | Size | Role |
| :--- | :--- | :--- | :--- |
| **MohlerASAG** | Benchmark Short Answer Grading dataset. | ~2,300 pairs | Training & Testing (Real World) |
| **GenAI Augmented** | Custom synthetic dataset focusing on "Hard Negatives". | ~1,000+ pairs | Training (Robustness) |

---

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/beyond-grading.git](https://github.com/your-username/beyond-grading.git)
    cd beyond-grading
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **(Optional) Set up OpenAI API key** (for data generation):
    ```bash
    export OPENAI_API_KEY='your-api-key-here'
    ```

---

##  Usage

### 1. Data Generation (Synthetic)
Generate new "Hard Negative" student answers using the LLM agent:
```bash
python src/data_generation.py --topic "Data Structures" --count 500
