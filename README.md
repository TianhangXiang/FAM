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
* **OS:** [e.g., Ubuntu 20.04]
* **Python:** [e.g., 3.10]
* **CUDA:** [e.g., 11.8 / 12.1]

Since our codebase is built upon VLM2Vec, setting up the environment is very straightforward.

### Option 1: Reuse VLM2Vec Environment (Fastest)
If you have already configured the environment for [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec), you can seamlessly reuse it without installing any extra packages. Simply activate your existing environment:

```bash
conda activate [your_vlm2vec_env_name]


### Option 2: Install from Scratch
If you are setting this up for the first time, please follow the steps below to create a new environment:

git clone [https://github.com/TianhangXiang/FAM.git](https://github.com/TianhangXiang/FAM.git)
cd FAM

conda create -n FAM python=3.10 -y
conda activate FAM

pip install -r requirements.txt


---

## 📦 Dataset Preparation

**1. Download the dataset**
Please download the required dataset from [Link/Source].

**2. Directory Structure**
Extract and place the dataset into the `data/` directory. The expected structure is as follows:

```text
data/
├── train/
│   ├── class1/
│   └── class2/
├── val/
└── test/
```

**3. Data Preprocessing (Optional)**
If extra data cleaning or formatting is needed, run the following command:
```bash
python scripts/preprocess.py --data_dir ./data --output_dir ./data_processed
```

---

## 🚀 Training

Once the environment and dataset are ready, you can start training the model. 

**Basic Training Command:**
```bash
python train.py --config configs/default.yaml
```

**Common Arguments:**
* `--batch_size`: Batch size (default: 32)
* `--epochs`: Number of training epochs (default: 100)
* `--lr`: Learning rate (default: 1e-4)
* `--resume`: Path to a checkpoint to resume training

Logs and model weights will be automatically saved in the `output/checkpoints/` directory during training.

---

## 📊 Evaluation

Use the trained weights to evaluate the model on the test set and get the final metrics.

**Evaluation Command:**
```bash
python eval.py --weights output/checkpoints/best_model.pth --data_dir ./data/test
```

**Expected Output Example:**
```text
Test Accuracy: 95.4%
Precision: 0.94
Recall: 0.96
```

---

## 📝 TODO

- [x] Open-source the core code for MAC and VEIN.
- [x] Release the training and evaluation scripts.
- [ ] Refactor the codebase to facilitate easier training and better reproducibility.
- [ ] Release the pipeline for Qwen2-VL.
---

## Acknowledgement
Our code is mainly based on [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec), thanks for their great contribution!
