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
* **Python:** [e.g., 3.8+]
* **CUDA:** [e.g., 11.6]

**1. Clone the repository**
```bash
git clone https://github.com/TianhangXiang/FAM.git
cd FAM
```

**2. Create a virtual environment (Recommended)**
```bash
conda create -n myenv python=3.8 -y
conda activate myenv
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

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

Future plans and improvements for this project:

- [x] Implement basic training and evaluation pipeline
- [x] Add support for [e.g., Base model name]
- [ ] Add Multi-GPU Distributed Data Parallel (DDP) support
- [ ] Integrate TensorBoard / Weights & Biases (W&B) for logging
- [ ] Add more data augmentation strategies
- [ ] Release pre-trained weights

---
*Feel free to open an Issue or submit a Pull Request!*
