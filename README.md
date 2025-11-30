# Beyond Grading: Diagnostic Misconception Analysis via GenAI 🎓🤖

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![HuggingFace](https://img.shields.io/badge/Transformers-4.30-yellow)
![Status](https://img.shields.io/badge/Status-In%20Progress-green)

> **Course Project:** Deep Learning & NLP | **Focus:** Educational Data Mining & Synthetic Data Generation

---

## [cite_start]📖 Project Motivation [cite: 257]
In modern education, especially in large-scale environments like MOOCs, providing personalized feedback on open-ended questions is a significant bottleneck. Standard automated systems often rely on keyword matching or simple binary grading (Correct/Incorrect), failing to address **why** a student made a mistake.
This project aims to shift from **Grading** to **Diagnosis**, leveraging Generative AI to identify specific conceptual gaps in student responses.

## [cite_start]🎯 Problem Statement [cite: 258]
The goal is to develop an NLP classification model that analyzes free-text student answers in STEM domains and classifies them into diagnostic categories based on conceptual understanding. Unlike standard sentiment analysis, this task requires deep semantic alignment with expert knowledge.

## [cite_start]🖼️ Visual Abstract [cite: 259]
*(Place your pipeline diagram here - e.g., image of Data Generation -> Processing -> Model -> Output)*
![Pipeline Diagram](assets/pipeline_diagram.png)

## [cite_start]📂 Datasets [cite: 260]
We utilize a hybrid dataset strategy combining real-world educational data with synthetic augmentation:
1.  **Real Data (Source):** [SciEntsBank](https://github.com/SebOchs/SciEntsBank) - A benchmark dataset containing expert-labeled student answers to science questions.
2.  **Synthetic Data (Augmented):** A custom dataset generated using LLMs to address class imbalance and provide examples of specific "Hard Negative" misconceptions.

## [cite_start]🧬 Data Augmentation & Generation [cite: 261-262]
A core novelty of this project is the **GenAI-driven data strategy**.
* **Method:** We use **GPT-4o / Llama-3** with few-shot prompting.
* **Strategy:** Conditional Generation. We prompt the LLM to act as a student with a specific misconception (e.g., *"Confuse mass with weight"*) and generate a plausible but incorrect answer.
* **Goal:** To balance the dataset (which naturally skews towards "Correct" answers) and improve the model's ability to detect subtle logic errors.

## [cite_start]🔀 Input / Output [cite: 263]
* **Input:**
    * `Question Context`
    * `Reference Answer` (Expert answer)
    * `Student Response`
* **Output:**
    * **Label:** `Correct`, `Partially Correct`, or `Misconception`
    * **Confidence Score:** (0.0 - 1.0)

## [cite_start]📝 Examples [cite: 264]

| Question | Student Answer | Prediction |
| :--- | :--- | :--- |
| *Explain why the object falls.* | *"Because gravity pushes it down."* | **Misconception** (Gravity pulls, doesn't push) |
| *What is a variable?* | *"It is a box that holds data."* | **Partially Correct** (Metaphorical, lacks precision) |
| *Define acceleration.* | *"Rate of change of velocity."* | **Correct** |

## [cite_start]🧠 Models and Pipelines [cite: 265]
We implement a **Cross-Encoder** architecture for semantic similarity and classification:
1.  **Preprocessing:** Concatenation of [Question + Reference + Student Answer].
2.  **Backbone:** Fine-tuning Transformer-based models (e.g., `microsoft/deberta-v3-base` or `sentence-transformers`).
3.  **Baseline:** Comparison against Zero-Shot LLM classification and TF-IDF logistic regression.

## [cite_start]⚙️ Training Process [cite: 266]
* **Loss Function:** Weighted Cross-Entropy Loss (to handle class imbalance).
* **Optimizer:** AdamW.
* **Strategy:** First training on the synthetic dataset (Pre-training), followed by fine-tuning on the real SciEntsBank data.

## [cite_start]📊 Metrics [cite: 267]
Since the data is imbalanced (fewer misconceptions than correct answers), we prioritize:
* **Macro-Averaged F1-Score** (Primary KPI)
* **Confusion Matrix** (To analyze Partial vs. Misconception errors)

## [cite_start]📈 Results [cite: 268]
*(To be updated after the final training phase. Currently presenting Baseline results).*

| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| Zero-Shot Baseline | TBD | TBD |
| **Our Model (DeBERTa)** | **TBD** | **TBD** |

## [cite_start]📂 Repository Structure [cite: 269]
```bash
├── data/                  # Raw and processed datasets
│   ├── real/              # SciEntsBank data
│   └── synthetic/         # Generated GenAI data
├── notebooks/             # Jupyter Notebooks for EDA and Training
├── src/                   # Source code
│   ├── data_gen.py        # Script for LLM data generation
│   ├── train.py           # Training loop
│   └── inference.py       # Prediction script
├── results/               # Confusion matrices and logs
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
