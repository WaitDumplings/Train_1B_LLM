from datasets import load_dataset, Dataset
import argparse
import os

def main(download=True, to_parquet=False, parquet_path_list=None, save_path="./"):
    if parquet_path_list is None:
        parquet_path_list = []

    if download:
        # Download Wikipedia Datasets (EN)
        print("Downloading Wikipedia dataset...")
        load_dataset("carlosejimenez/wikipedia-20220301.en-0.005-validation")
        
        # Download Gutenberg (EN)
        print("Downloading Gutenberg dataset...")
        load_dataset("sedthh/gutenberg_english")
        
        # Download Books3
        print("Downloading Books3 dataset...")
        load_dataset("SaylorTwift/the_pile_books3_minus_gutenberg")

    if to_parquet:
        # Ensure the save directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Convert Arrow files to Parquet
        for path in parquet_path_list:
            dataset_name = os.path.basename(path)

            arrow_files = [f for f in os.listdir(path) if f.lower().endswith(".arrow")]
            for arrow_file in arrow_files:
                dataset = Dataset.from_file(os.path.join(path, arrow_file))
                
                output_name = os.path.splitext(arrow_file)[0] + ".parquet"
                dataset_save_path = os.path.join(save_path, dataset_name)
                
                if not os.path.exists(dataset_save_path):
                    os.makedirs(dataset_save_path)
                
                parquet_filename = os.path.join(dataset_save_path, output_name)
                dataset.to_parquet(parquet_filename)
            
            print(f"Dataset saved as Parquet at: {dataset_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and convert datasets.")
    parser.add_argument("--download", action="store_true", help="Download datasets")
    parser.add_argument("--to_parquet", action="store_true", help="Convert datasets to Parquet format")
    parser.add_argument("--parquet_path_list", nargs="+", default=[], help="Paths to Arrow files for conversion")
    parser.add_argument("--save_path", default="./", help="Path to save Parquet files")
    
    args = parser.parse_args()
    
    main(
        download=args.download,
        to_parquet=args.to_parquet,
        parquet_path_list=args.parquet_path_list,
        save_path=args.save_path,
    )
