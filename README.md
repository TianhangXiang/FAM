# FAM

The official pytorch implementation of FAM: Fine-grained Alignment Matters in Multimodal Embedding Learning with Large Vision-Language Models

## Table of Contents
- [Environment Preparation](#️-environment-preparation)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [TODO](#-todo)

---

## 🛠️ Environment Preparation

This project was developed and tested under the following environment:
* **OS:** Ubuntu 20.04
* **Python:** 3.10
* **CUDA:** 11.8
* **Pytorc:** 2.1.1
* **transformers:** 4.49.0
Since our codebase is built upon VLM2Vec, setting up the environment is very straightforward.

### Option 1: Reuse VLM2Vec Environment (Fastest)
If you have already configured the environment for [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec), you can seamlessly reuse it without installing any extra packages. Simply activate your existing environment:

```bash
conda activate [your_vlm2vec_env_name]
```

### Option 2: Install from Scratch
If you are setting this up for the first time, please follow the steps below to create a new environment:
```bash
git clone [https://github.com/TianhangXiang/FAM.git](https://github.com/TianhangXiang/FAM.git)
cd FAM

conda create -n FAM python=3.10 -y
conda activate FAM

pip install -r requirements.txt
```


---

## 📦 Dataset Preparation

**1.Download the pretrain data for MAC (LLAVA)**


**2.Download the MMEB data**
Please download the required dataset from [Link/Source].

**3.Organize the training data**
Extract and place the dataset into the `data/` directory. The expected structure is as follows:

```text
data/
├── train/
│   ├── class1/
│   └── class2/
├── val/
└── test/
```

**3. Data Preprocessing**
[TODO]

---

## 🚀 Training
[TODO]

---

## 📊 Evaluation

[TODO]

---

## 📝 TODO

- [x] Open-source the core code for MAC and VEIN.
- [x] Release the training and evaluation scripts.
- [ ] Release the data preprocess code.
- [ ] Refactor the codebase to facilitate easier training and better reproducibility.
- [ ] Release the pipeline for Qwen serises.
---

## Acknowledgement
Our code is mainly based on [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec), thanks for their great contribution!
