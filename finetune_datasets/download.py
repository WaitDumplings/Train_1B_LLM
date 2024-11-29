from datasets import load_dataset, Dataset
import argparse
import os

def main(download=True, to_parquet=False, parquet_path_list=[], save_path="./data_parquet", cache_dir = "./data_cache"):

    datasetnames = ["naturalquestionsshortqa", "python_code_instructions", "causal-lm", "co_t-collection", "la_mini-instruction", "math_instruct", "math_copy_n1"]
    if len(parquet_path_list) == 0 and to_parquet:
        # default path for wikipedia dataset, gutenberg dataset and books3 dataset
        parquet_path_list = ["./data_cache/causal-lm___instructions/default/0.0.0/7784bfb5f6dc52b4f861ac10e09576769e8415c9",
                             "./data_cache/iamtarun___python_code_instructions_18k_alpaca/default/0.0.0/7cae181e29701a8663a07a3ea43c8e105b663ba1",
                             "./data_cache/kaist-ai___co_t-collection/en/1.0.0/00cb478a00ef346084c6325b38624b4b5fd9ed98d8d6010e66bdb48bd058f8c4",
                             "./data_cache/lucadiliello___naturalquestionsshortqa/default/0.0.0/49435450b0abb287be45639fdaa49a9b56512f9b",
                             "./data_cache/MBZUAI___la_mini-instruction/default/0.0.0/7372b3c04dd7a09e4ca5ae572557d843cb4b5482",
                             "./data_cache/TIGER-Lab___math_instruct/default/0.0.0/b4fdc323a7be1379c9c7c0b67b1de72dfee2111a",
                             "./data_cache/altas-ai___math_copy_n1/default/0.0.0/dd71deedca687e430256981a9d5d6f274624f5a5"]
    

    if len(parquet_path_list) == 0 and to_parquet:
        raise ValueError("parquet_path_list is empty!")

    if download:
        # Download CoT
        print("Downloading CoT dataset...")
        ds = load_dataset("kaist-ai/CoT-Collection", cache_dir=cache_dir)
        print(f"Data size is {ds.shape['train'][0]}")
        
        # Download MBZUAI/LaMini-instruction (EN)
        print("Downloading LaMini dataset...")
        ds = load_dataset("MBZUAI/LaMini-instruction", cache_dir=cache_dir)
        print(f"Data size is {ds.shape['train'][0]}")   
        
        # Download Math
        print("Downloading Math dataset...")
        ds = load_dataset("altas-ai/MATH_copy_n1", cache_dir=cache_dir)
        print(f"Data size is {ds.shape['train'][0]}")   

        # Download Math Instruct
        print("Downloading Math Instruct dataset...")
        ds = load_dataset("TIGER-Lab/MathInstruct", cache_dir=cache_dir)
        print(f"Data size is {ds.shape['train'][0]}")  

        # Download Casual Instruct
        print("Downloading Causal LM Instruct dataset...")
        ds = load_dataset("causal-lm/instructions", cache_dir=cache_dir) 
        print(f"Data size is {ds.shape['train'][0]}")  

        # Download Python Instuct
        print("Downloading Python Instruct dataset...")
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", cache_dir=cache_dir)
        print(f"Data size is {ds.shape['train'][0]}")  

        # Download natural QA
        print("Downloading Natural QA dataset...")
        ds = load_dataset("lucadiliello/naturalquestionsshortqa", cache_dir=cache_dir)
        print(f"Data size is {ds.shape['train'][0]}")  

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
