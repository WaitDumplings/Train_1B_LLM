# Train 1B-LLM From Scratch

## Introduction

This project provides a comprehensive pipeline to train a 1B-parameter large language model (LLM) from scratch. The steps include dataset preparation, tokenizer training, instruction fine-tuning, and evaluation on MMLU benchmarks.

---

## Steps

### Step 1: Data Download
- Use `./datasets/download.py` to download datasets.
- The datasets are downloaded in `.arrow` format, with C4 data in `.json.zip` format.
- Convert all datasets to `.txt` format for streamlined training.
- Tokenizers are trained using the Huggingface `tokenizer` package with the Byte Pair Encoding (BPE) method.

---

### Step 2: Data Format Conversion
1. **arrow -> parquet**:
   - Set `to_parquet=True` in `./datasets/download.py`.
   - Specify the dataset paths in `parquet_path_list`.
   - Example configuration for Wikipedia data:
     ```python
     parquet_path_list = [~/.cache/huggingface/datasets/carlosejimenez___wikipedia-20220301.en-0.005-validation/default/0.0.0/c6abeb0e16beb83f93433d09890dfa7bc2dff1ff/]
     ```

2. **parquet -> txt**:
   - Use `./datasets/dataset_processor.py` to convert parquet files to `.txt` format.
   - Supported datasets: C4, Gutenberg, Books3, and Wikipedia (EN).
   - For other datasets, implement custom data cleaning before converting to `.txt`.
   - Specify dataset paths and names in the script.
   - Converted `.txt` files will be saved in the `./datasets/data_txt` directory.
   - For `C4` dataset, you need to download (via git lfs) and put  `c4\en` to the path `./datasets/data_parquet` to do the transformation.
---

### Step 3: Tokenizer Training
- Once `.txt` format datasets are prepared, train the tokenizer using `train_tokenizer.py`.
- Configure the following parameters:
  - `model_root`: Root directory for the tokenizer.
  - `tokenizer`: Folder to store the tokenizer.
  - `tokenizer_name`: Name of the tokenizer.
  - `data_root`: Root directory of the datasets.
  - `target_dirs`: Names of the folders containing `.txt` files in the dataset root directory.

---

### Step 4: Instruction Fine-Tuning
1. **Dataset Preparation**:
   - Navigate to `./fine-tune_datasets` and run:
     - `download.py`: Download the fine-tuning datasets.
     - `finetune_dataset_preprocess.py`: Process the datasets (similar to Steps 1 and 2).
   - Processed datasets will be saved in the appropriate format.

2. **Fine-Tuning**:
   - Run `./finetune.py` for instruction fine-tuning.
   - Ensure the model path and tokenizer path are correctly configured in the script.

---

### Step 5: MMLU Evaluation
1. **Dataset Preparation**:
   - Navigate to `./evaluate_datasets` and execute:
     - `download.py`: Download the MMLU datasets.
     - `mmlu_preprocess.py`: Process the datasets. Processed data will be stored in the `./evaluate_datasets/data_txt` folder.

2. **Evaluation**:
   - Run `./evaluate.py` to evaluate the fine-tuned model on MMLU.
   - Ensure the model path and tokenizer path are correctly configured.

---
# Project Directory Structure

The directory structure of this project is as follows:

```plaintext
.
├── ckpt                      # Stores the checkpoints generated during training and fine-tuning.
├── datasets
│   ├── dataset_processor.py  # Handles dataset processing, including format conversion and cleaning.
│   └── download.py           # Downloads raw datasets for training.
├── fine-tune_datasets
│   ├── download.py           # Downloads fine-tuning datasets.
│   └── finetune_dataset_preprocess.py  # Preprocesses the datasets for fine-tuning.
├── evaluate_datasets
│   ├── download.py           # Downloads evaluation datasets (e.g., MMLU).
│   └── mmlu_preprocess.py    # Preprocesses MMLU datasets for evaluation.
├── tokenizer
│   └── tokenizer_new.json    # Configuration and vocabulary for the trained tokenizer.
├── README.md                 # Documentation for the project, including setup instructions and usage guidelines.
├── main.py                   # The primary entry point for the project, typically used for initializing workflows like training or fine-tuning.
├── model.py                  # Defines the model architecture and implementation details for the 1B-parameter LLM.
├── train_tokenizer.py        # Script for training the tokenizer using the preprocessed `.txt` datasets.
├── finetune.py               # Script for performing instruction fine-tuning on the prepared datasets.
├── dataloader.py             # Responsible for data loading and batching during training, fine-tuning, or evaluation.
├── configuration.py          # Contains configuration settings, including hyperparameters, model paths, and training parameters.
├── chat_gradio.py            # Deploys a Gradio interface to interact with the trained model in a chat-like format.
└── evaluate.py               # Evaluates the fine-tuned model on benchmark datasets, such as MMLU.

