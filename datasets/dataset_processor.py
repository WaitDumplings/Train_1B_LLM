import os
import re
import gzip
import json
import pandas as pd
from tqdm import tqdm
from langdetect import detect_langs


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


def convert_parquet_to_txt(
    root, source_dir, target_dir, column_name="text", process_func=None
):
    """
    Convert Parquet files in a source directory to text files in a target directory.
    """
    source_path = os.path.join(root, source_dir)
    target_path = create_target_dir(root, target_dir)
    files = list_files_with_extension(source_path, ".parquet")

    process_name = source_dir.split("/")[0]
    for file in tqdm(files, desc=f"Processing {process_name}"):
        data = pd.read_parquet(os.path.join(source_path, file))[column_name]

        if process_func:
            data = data.apply(process_func)

        output_file = os.path.join(target_path, file.replace(".parquet", ".txt"))
        if os.path.exists(output_file):
            os.remove(output_file)

        with open(output_file, "w") as f:
            f.writelines(data)


def filter_books3(root, source_dir, target_dir, gutenberg_list):
    """
    Filter and save Books3 dataset based on language and uniqueness criteria.
    """
    source_path = os.path.join(root, source_dir)
    target_path = create_target_dir(root, target_dir)
    files = list_files_with_extension(source_path, ".parquet")

    process_name = source_dir.split("/")[-1]
    for file in tqdm(files, desc=f"Processing {process_name}"):
        path = os.path.join(source_path, file)
        books = pd.read_parquet(path)
        original_count = books.shape[0]

        filter_index = books.apply(detect_en, axis=1, args=(gutenberg_list,))
        books_filtered = books[filter_index]
        filtered_count = books_filtered.shape[0]

        print(
            f"{file}: original: {original_count}, filtered: {filtered_count}, "
            f"ratio: {filtered_count/original_count:.2f}"
        )

        data = books_filtered["text"].to_list()
        output_file = os.path.join(target_path, file.replace(".parquet", ".txt"))
        if os.path.exists(output_file):
            os.remove(output_file)

        with open(output_file, "w") as f:
            for book in data:
                book = book.replace("\n\n", "\n")
                book = [
                    line
                    for line in book.split("\n")
                    if not re.match(r"^ *[|0-9.-]+ *$", line)
                ]
                f.write("\n".join(book) + "\r\n\r\n\r\n")


def convert_c4_to_txt(root, source_dir, target_dir):
    """
    Convert C4 JSON.gz files to text files.
    """
    source_path = os.path.join(root, source_dir)
    target_path = create_target_dir(root, target_dir)
    files = list_files_with_extension(source_path, ".json.gz")

    process_name = source_dir.split("/")[-1]
    for file in tqdm(files, desc=f"Processing {process_name}"):
        with gzip.open(os.path.join(source_path, file), "r") as f:
            json_str = f.read().decode("utf-8")
            json_list = json_str[1:-2].split("}\n{")
            output_file = os.path.join(target_path, file.replace(".json.gz", ".txt"))
            if os.path.exists(output_file):
                os.remove(output_file)

            with open(output_file, "w") as tf:
                for data in json_list:
                    data = json.loads("{" + data + "}")
                    if "placeholder page" in data.get("text", ""):
                        continue
                    tf.write(data["text"] + "\r\n\r\n\r\n")


def detect_en(x, gutenberg_list):
    """
    Detect if a book is in English and not in the Gutenberg list.
    """
    title, text = x["title"], x["text"]
    if title in gutenberg_list:
        return False
    if len(text) < 1000:
        return False
    langs_list = detect_langs(text[:10000])
    return langs_list[0].lang == "en" and langs_list[0].prob >= 0.8


if __name__ == "__main__":
    root = "./"
 
    # Process Wikipedia dataset
    convert_parquet_to_txt(
        root,
        "wikipedia",
        "english_wikipedia",
        column_name="text",
        process_func=lambda x: x + "\r\n\r\n\r\n",
    )

    # Process Gutenberg dataset
    convert_parquet_to_txt(
        root,
        "gutenberg_english",
        "english_gutenberg",
        column_name="TEXT",
        process_func=lambda book: "\r\n\r\n\r\n".join(
            [
                "".join(
                    map(
                        lambda x: x + " " if x else "\n",
                        map(str.strip, filter(bool, book.split("\r\n"))),
                    )
                )
            ]
        ),
    )

    # Filter and process Books3 dataset
    gutenberg_list = set()
    for file in list_files_with_extension(os.path.join(root, "gutenberg_english"), ".parquet"):
        books = pd.read_parquet(os.path.join(root, "gutenberg_english", file))
        gutenberg_list.update(
            books["METADATA"].apply(lambda x: json.loads(x)["title"]).tolist()
        )
    filter_books3(root, "books3", "english_books3", gutenberg_list)

    # Process C4 dataset
    convert_c4_to_txt(root, "c4/en", "english_c4")

