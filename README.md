# Train 1B-LLM From Scratch

## Introduction

This project aims to train a 1B-parameter large language model (LLM) from scratch. Below are the key functionalities implemented:

1. **Data Download**
   - Use `./datasets/download.py` to download datasets.
   - The downloaded data is in `.arrow` format (C4 data is in `.json.zip` format).
   - To streamline the training process, all data is converted to `.txt` format.
   - Utilize the Huggingface `tokenizer` package to train tokenizers with the Byte Pair Encoding (BPE) method.

2. **Data Format Conversion**
   - **arrow -> parquet**:
     - Set `to_parquet=True` in `./datasets/download.py`.
     - Specify the paths of datasets to convert in `parquet_path_list`.
     - Example: If Wikipedia data is downloaded and stored in Huggingface's cache directory `~/.cache/huggingface/datasets`, configure:
       ```python
       parquet_path_list = [~/.cache/huggingface/datasets/carlosejimenez___wikipedia-20220301.en-0.005-validation/default/0.0.0/c6abeb0e16beb83f93433d09890dfa7bc2dff1ff/]
       ```

   - **parquet -> txt**:
     - Implemented in `./datasets/dataset_processor.py`.
     - Currently supports conversion for C4, Gutenberg, Books3, and Wikipedia (EN) datasets.
     - For other datasets, you can perform custom data cleaning and convert them to `.txt` format using the same approach.
     - Simply specify the dataset paths and names.

3. **Training the Tokenizer**
   - Once `.txt` format datasets are prepared, use `train_tokenizer.py` to train a tokenizer.
   - The following parameters need to be configured:
     - `model_root`: Root directory for the tokenizer.
     - `tokenizer`: Folder to store the tokenizer.
     - `tokenizer_name`: Name of the tokenizer.
     - `data_root`: Root directory of the datasets.
     - `target_dirs`: Names of the folders containing `.txt` files in the dataset root directory.

## Project Directory Structure

```plaintext
.
├── ckpt
├── datasets
│   ├── dataset_processor.py
│   └── download.py
├── README.md
├── tokenizers
└── train_tokenizer.py
