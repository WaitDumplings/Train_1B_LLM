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
        "Wikipedia": "carlosejimenez/wikipedia-20220301.en-0.005-validation",
        "Gutenberg": "sedthh/gutenberg_english",
        "Books3": "SaylorTwift/the_pile_books3_minus_gutenberg",
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
        
        # Determine the dataset name based on the path
        for name in datasetnames:
            if name in path:
                dataset_name = name
                break
        
        if dataset_name is None:
            print(f"Skipping path {path}: no matching dataset name found.")
            continue

        # Process Arrow files in the given path
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


def main(download=True, to_parquet=False, parquet_path_list=None, cache_dir="./data_cache", save_path="./data_parquet"):
    """
    Main function to download datasets and/or convert Arrow files to Parquet.

    Args:
        download (bool): Whether to download datasets.
        to_parquet (bool): Whether to convert datasets to Parquet format.
        parquet_path_list (list): List of paths to Arrow files for conversion.
        save_path (str): Directory to save converted files.
    """
    datasetnames = ["wikipedia",
                    "gutenberg_english",
                    "books3"]
    
    if parquet_path_list is None:
        parquet_path_list = []
    
    # Default paths for datasets if no paths are provided
    if not parquet_path_list and to_parquet:
        parquet_path_list = [
            "./data_cache/carlosejimenez___wikipedia-20220301.en-0.005-validation/default/0.0.0/c6abeb0e16beb83f93433d09890dfa7bc2dff1ff",
            "./data_cache/SaylorTwift___the_pile_books3_minus_gutenberg/default/0.0.0/54b79259e19ef0a7a456b6436a63470c63ba5b0f",
            "./data_cache/sedthh___gutenberg_english/default/0.0.0/28973b04f28fd7be4a6186a042bc26159d4366ca"
        ]

    # Check if paths are empty when Parquet conversion is enabled
    if not parquet_path_list and to_parquet:
        raise ValueError("parquet_path_list is empty! Provide valid paths for conversion.")

    # Download datasets if required
    if download:
        download_datasets(cache_dir)
    breakpoint()
    # Convert Arrow files to Parquet if required
    if to_parquet:
        convert_to_parquet(parquet_path_list, save_path, datasetnames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and/or process datasets.")
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
