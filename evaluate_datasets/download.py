from datasets import load_dataset, Dataset
import argparse
import os

def main(download=True, to_parquet=False, parquet_path_list=[], save_path="./data_parquet", cache_dir = "./data_cache"):

    datasetnames = ["mmlu"]
    if len(parquet_path_list) == 0 and to_parquet:
        # default path for wikipedia dataset, gutenberg dataset and books3 dataset
        parquet_path_list = ["./data_cache/lighteval___mmlu/all/1.0.0/e24764f1fb58c26b5f622157644f2e5fe77e5b01"]
    

    if len(parquet_path_list) == 0 and to_parquet:
        raise ValueError("parquet_path_list is empty!")

    if download:
        # Download MMLU
        print("Downloading MMLU dataset...")
        ds = load_dataset("lighteval/mmlu", "all", cache_dir=cache_dir)
        print(f"Data size is {ds.shape['auxiliary_train'][0]}")

    if to_parquet:
        dataset_name = None

        # Ensure the save directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Convert Arrow files to Parquet
        for path in parquet_path_list:

            # Set dataset names   
            for name in datasetnames:
                if name in path:
                    dataset_name = name
                    break
                
            if dataset_name is None:
                continue

            arrow_files = [f for f in os.listdir(path) if f.lower().endswith(".arrow")]
            for arrow_file in arrow_files:
                dataset = Dataset.from_file(os.path.join(path, arrow_file))
                
                output_name = os.path.splitext(arrow_file)[0] + ".parquet"
                dataset_save_path = os.path.join(save_path, dataset_name)

                if not os.path.exists(dataset_save_path):
                    os.makedirs(dataset_save_path)
                
                parquet_filename = os.path.join(dataset_save_path, output_name)
                dataset.to_parquet(parquet_filename)
            
            print(f"{dataset_name} dataset saved as Parquet at: {dataset_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and convert datasets.")
    parser.add_argument("--download", default=False, help="Download datasets")
    parser.add_argument("--to_parquet", default=False, help="Convert datasets to Parquet format")
    parser.add_argument("--parquet_path_list", nargs="+", default=[], help="Paths to Arrow files for conversion")
    parser.add_argument("--save_path", default="./data_parquet", help="Path to save Parquet files")
    
    args = parser.parse_args()
    
    main(
        download=args.download,
        to_parquet=args.to_parquet,
        parquet_path_list=args.parquet_path_list,
        save_path=args.save_path,
    )
