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
    target_path = create_target_dir(root, save_path)

    files = list_files_with_extension(source_path, ".parquet")
    all_data = []

    # Process each file
    for file in tqdm(files, desc="Processing {} Dataset".format(name)):

        # Load parquet file and extract 'source' and 'rationale' columns
        data = pd.read_parquet(os.path.join(source_path, file))[column_name]
        
        if preprocess:
            data = preprocess(data)
            
        if len(column_name) == 4:
            # Format rows into the desired template
            formatted_data = data.apply(
                lambda row: (
                    template.format(
                        instruction=row[column_name[0]],
                        prompt=row[column_name[1]],
                        input=row[column_name[2]] if row[column_name[2]] else "",  # Only include input if it's not empty
                        output=row[column_name[3]]
                    ) if len(row[column_name[2]])>0 else 
                    template.replace("{input}\n", "").format(
                        instruction=row[column_name[0]],
                        prompt=row[column_name[1]],
                        output=row[column_name[3]]
                    )
                ) + spliter,
                axis=1
            )

        elif len(column_name) == 3:
            # Format rows into the desired template
            formatted_data = data.apply(
                lambda row: template.format(instruction=row[column_name[0]], input=row[column_name[1]], output=row[column_name[2]]) + spliter,
                axis=1
            )

        elif len(column_name) == 2:
            # Format rows into the desired template
            formatted_data = data.apply(
                lambda row: template.format(input=row[column_name[0]], output=row[column_name[1]]) + spliter,
                axis=1
            )
        else:
            raise ValueError("the input column is larger than 4!")

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

if __name__ == "__main__":
    root = "./"
    
    # casual_lm
    template_casuallm = (
        "Input: {instruction}\n"
        "{input}\n"
        "Response: {output}"
    )

    Instruct_parquet_save_to_txt(root, 
                                 source = "./data_parquet/causal-lm", 
                                 save_path = "./data_txt/instruct_en_casuallm", 
                                 column_name = ["instruction", "input", "output"], 
                                 partition = 1024, 
                                 spliter = '\n▁\n▁\n', 
                                 template = template_casuallm, 
                                 name = "CausalLM"
                                 )

    # co_t
    template_cot = (
        "Input: {input}\n"
        "Response: {output}"
    )

    Instruct_parquet_save_to_txt(root, 
                                 source = "./data_parquet/co_t-collection",
                                 save_path = "./data_txt/instruct_en_CoT", 
                                 column_name = ["source", "rationale"], 
                                 partition = 1024, 
                                 spliter = '\n▁\n▁\n', 
                                 template = template_cot, 
                                 name = "CoT"
                                 )

    # la_mini
    template_la_mini = (
        "Input: {input}\n"
        "Response: {output}"
    )

    Instruct_parquet_save_to_txt(root, 
                                 source = "./data_parquet/la_mini-instruction",
                                 save_path = "./data_txt/instruct_en_LAmini", 
                                 column_name = ["instruction", "response"], 
                                 partition = 1024, 
                                 spliter = '\n▁\n▁\n', 
                                 template = template_la_mini, 
                                 name = "LAmini"
                                 )

    # math_copy_n1
    template_math_n1 = (
        "Input: {input}\n"
        "Response: {output}"
    )

    Instruct_parquet_save_to_txt(root, 
                                 source = "./data_parquet/math_copy_n1",
                                 save_path = "./data_txt/instruct_en_MATHn1", 
                                 column_name = ["problem", "solution"], 
                                 partition = 1024, 
                                 spliter = '\n▁\n▁\n', 
                                 template = template_math_n1, 
                                 name = "Mathn1"
                                 )

    # naturalquestion
    template_natural_qa = (
        "Input: {instruction}\n"
        "{input}\n"
        "Response: {output}"
    )

    Instruct_parquet_save_to_txt(root, 
                                 source = "./data_parquet/naturalquestionsshortqa",
                                 save_path = "./data_txt/instruct_en_NaturalQA", 
                                 column_name = ["context", "question", "answers"], 
                                 partition = 1024, 
                                 spliter = '\n▁\n▁\n', 
                                 template = template_natural_qa, 
                                 name = "NaturalQA",
                                 preprocess = preprocess_natural_qa
                                 )

    # python_code
    template_python_code = (
        "Input: {instruction}\n"
        "{prompt}\n"
        "{input}\n"
        "Response: {output}"
    )

    Instruct_parquet_save_to_txt(root, 
                                 source = "./data_parquet/python_code_instructions",
                                 save_path = "./data_txt/instruct_en_Python", 
                                 column_name = ["instruction", "prompt", "input", "output"], 
                                 partition = 1024, 
                                 spliter = '\n▁\n▁\n', 
                                 template = template_python_code, 
                                 name = "Python Instruction",
                                 )

    # python_code
    template_math_instruct = (
        "Input: {input}\n"
        "Response: {output}"
    )

    Instruct_parquet_save_to_txt(root, 
                                 source = "./data_parquet/math_instruct",
                                 save_path = "./data_txt/instruct_en_math_Instruct", 
                                 column_name = ["instruction", "output"], 
                                 partition = 1024, 
                                 spliter = '\n▁\n▁\n', 
                                 template = template_math_instruct, 
                                 name = "Math Instruction",
                                 )
    
