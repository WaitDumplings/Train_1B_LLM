import os
import re
import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset


def preprocess_natural_qa(data):
    data["context"] = (
        data["context"]
        .str.replace(r"^<P>|</P>$", "", regex=True)  # 移除 <P> 和 </P>
        .str.strip()  # remove blank
    )
    return data

def create_target_dir(root, target_dir):
    """
    Ensure the target directory exists.
    """
    path = os.path.join(root, target_dir)
    os.makedirs(path, exist_ok=True)
    return path


def list_files_with_extension(directory, extension):
    """
    List files in a directory with a specific extension.
    """
    return sorted([f for f in os.listdir(directory) if f.endswith(extension)])

def Instruct_parquet_save_to_txt(root, source, save_path, column_name, partition, spliter, template, name, preprocess = None):
    source_path = os.path.join(root, source)
    target_path = create_target_dir(root, save_path)  # Ensure target directory exists

    # List all .parquet files in the source directory
    files = list_files_with_extension(source_path, ".parquet")
    all_data = []

    # Process each file
    for file in tqdm(files, desc=f"Processing {name} Dataset"):
        # Load parquet file and extract the specified columns
        file_path = os.path.join(source_path, file)
        data = pd.read_parquet(file_path)

        # Ensure all required columns exist
        if not all(col in data.columns for col in column_name):
            raise ValueError(f"One or more specified columns are missing in {file_path}")

        # Extract choices into separate columns for formatting
        data[["A", "B", "C", "D"]] = data[column_name[1]].apply(
            lambda x: pd.Series(x[:4])  # Ensure only the first 4 elements are taken
        )

        # Answer 1,2,3,4 -> A B C D
        number_to_letter = {1: "A", 2: "B", 3: "C", 4: "D"}
        data[column_name[2]] = data[column_name[2]].map(number_to_letter)

        # Apply preprocessing if provided
        if preprocess:
            data = preprocess(data)

        # Format rows into the desired template
        formatted_data = data.apply(
            lambda row: template.format(
                question=row[column_name[0]],  # Question
                A=row["A"],  # Choice A
                B=row["B"],  # Choice B
                C=row["C"],  # Choice C
                D=row["D"],  # Choice D
                answer=row[column_name[2]]  # Correct answer
            ) + spliter,
            axis=1
        )

        all_data.extend(formatted_data.tolist())

    # Shuffle and partition the data
    knt = len(all_data)
    block_num = knt // partition + 1

    parts = np.arange(partition)[None, :]
    parts = np.tile(parts, (block_num, 1)).reshape(-1)
    np.random.shuffle(parts)
    parts = parts[:knt]

    print(f"Total entries: {knt}, Block number: {block_num}")

    # Save data into partitioned files
    for knt, text in enumerate(all_data):
        if knt % 100000 == 0:
            print(f"Processed {knt} entries")

        # Determine the file for this entry
        ti = parts[knt]
        tpath = os.path.join(save_path, f"train.{ti:04d}-of-{partition:04d}.txt")

        # Append the text to the file
        with open(tpath, 'a', encoding='utf-8') as wf:
            wf.write(text)
            
    return all_data

if __name__ == "__main__":
    root = "./"
    
    # mmlu
    template_mmlu = (
    "question: {question}\n"
    "choices: A.{A} B.{B} C.{C} D.{D}\n"
    "answer: {answer}"
        ) 

    Instruct_parquet_save_to_txt(root, 
                                 source = "./data_parquet/mmlu", 
                                 save_path = "./data_txt/evaluate_en_mmlu", 
                                 column_name = ["question", "choices", "answer"], 
                                 partition = 1024, 
                                 spliter = '\n▁\n▁\n', 
                                 template = template_mmlu, 
                                 name = "MMLU"
                                 )

   
