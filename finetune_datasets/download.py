from datasets import load_dataset, Dataset
import argparse
import os

def download_datasets(cache_dir):
    """
    Downloads specified datasets using the Hugging Face `datasets` library.

    Args:
        cache_dir (str): Directory to cache downloaded datasets.
    """
    datasets_to_download = {
        "CoT-Collection": "kaist-ai/CoT-Collection",
        "LaMini-Instruction": "MBZUAI/LaMini-instruction",
        "Math Copy N1": "altas-ai/MATH_copy_n1",
        "Math Instruct": "TIGER-Lab/MathInstruct",
        "Causal LM Instructions": "causal-lm/instructions",
        "Python Code Instructions": "iamtarun/python_code_instructions_18k_alpaca",
        "Natural Questions Short QA": "lucadiliello/naturalquestionsshortqa",
    }

    for name, path in datasets_to_download.items():
        print(f"Downloading {name} dataset...")
        ds = load_dataset(path, cache_dir=cache_dir)
        print(f"{name} dataset downloaded. Train set size: {ds['train'].num_rows}")


def convert_to_parquet(parquet_path_list, save_path, datasetnames):
    """
    Converts Arrow files to Parquet format and saves them to the specified directory.

    Args:
        parquet_path_list (list): List of paths to Arrow files.
        save_path (str): Directory to save the converted Parquet files.
        datasetnames (list): List of dataset names for labeling converted files.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for path in parquet_path_list:
        dataset_name = None

        # Match dataset name from path
        for name in datasetnames:
            if name in path:
                dataset_name = name
                break

        if dataset_name is None:
            print(f"Skipping path {path}: no matching dataset name found.")
            continue

        # Process Arrow files
        arrow_files = [f for f in os.listdir(path) if f.lower().endswith(".arrow")]
        for arrow_file in arrow_files:
            arrow_file_path = os.path.join(path, arrow_file)
            dataset = Dataset.from_file(arrow_file_path)

            # Prepare the output directory and file name
            dataset_save_path = os.path.join(save_path, dataset_name)
            if not os.path.exists(dataset_save_path):
                os.makedirs(dataset_save_path)
            
            output_name = os.path.splitext(arrow_file)[0] + ".parquet"
            parquet_filename = os.path.join(dataset_save_path, output_name)

            # Save as Parquet
            dataset.to_parquet(parquet_filename)
            print(f"{dataset_name} saved as Parquet at: {parquet_filename}")


def main(download=True, to_parquet=False, parquet_path_list=None, save_path="./data_parquet", cache_dir="./data_cache"):
    """
    Main function to download datasets and/or convert Arrow files to Parquet.

    Args:
        download (bool): Whether to download datasets.
        to_parquet (bool): Whether to convert datasets to Parquet format.
        parquet_path_list (list): List of paths to Arrow files for conversion.
        save_path (str): Directory to save converted files.
        cache_dir (str): Directory to cache downloaded datasets.
    """
    datasetnames = [
        "naturalquestionsshortqa", 
        "python_code_instructions", 
        "causal-lm", 
        "co_t-collection", 
        "la_mini-instruction", 
        "math_instruct", 
        "math_copy_n1"
    ]

    if parquet_path_list is None:
        parquet_path_list = []

    # Default paths for datasets if no paths are provided
    if not parquet_path_list and to_parquet:
        parquet_path_list = [
            "./data_cache/causal-lm___instructions/default/0.0.0/7784bfb5f6dc52b4f861ac10e09576769e8415c9",
            "./data_cache/iamtarun___python_code_instructions_18k_alpaca/default/0.0.0/7cae181e29701a8663a07a3ea43c8e105b663ba1",
            "./data_cache/kaist-ai___co_t-collection/en/1.0.0/00cb478a00ef346084c6325b38624b4b5fd9ed98d8d6010e66bdb48bd058f8c4",
            "./data_cache/lucadiliello___naturalquestionsshortqa/default/0.0.0/49435450b0abb287be45639fdaa49a9b56512f9b",
            "./data_cache/MBZUAI___la_mini-instruction/default/0.0.0/7372b3c04dd7a09e4ca5ae572557d843cb4b5482",
            "./data_cache/TIGER-Lab___math_instruct/default/0.0.0/b4fdc323a7be1379c9c7c0b67b1de72dfee2111a",
            "./data_cache/altas-ai___math_copy_n1/default/0.0.0/dd71deedca687e430256981a9d5d6f274624f5a5"
        ]

    # Check if paths are empty when Parquet conversion is enabled
    if not parquet_path_list and to_parquet:
        raise ValueError("parquet_path_list is empty! Provide valid paths for conversion.")

    # Download datasets if required
    if download:
        download_datasets(cache_dir)

    # Convert Arrow files to Parquet if required
    if to_parquet:
        convert_to_parquet(parquet_path_list, save_path, datasetnames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process datasets.")
    parser.add_argument("--download", default=True, help="Download datasets")
    parser.add_argument("--to_parquet", default=False, help="Convert datasets to Parquet format")
    parser.add_argument("--parquet_path_list", nargs="+", default=None, help="Paths to Arrow files for conversion")
    parser.add_argument("--save_path", default="./data_parquet", help="Directory to save Parquet files")
    parser.add_argument("--cache_dir", default="./data_cache", help="Directory to cache downloaded datasets")

    args = parser.parse_args()

    main(
        download=args.download,
        to_parquet=args.to_parquet,
        parquet_path_list=args.parquet_path_list,
        save_path=args.save_path,
        cache_dir=args.cache_dir
    )
