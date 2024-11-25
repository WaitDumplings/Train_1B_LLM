import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers import normalizers, pre_tokenizers
from tokenizers import decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
import json

def list_files(root_dir, sub_dirs, filter_keyword='train', sampling_ratio=1.0):
    """
    List files from target directories with optional filtering and sampling.

    Args:
        root_dir (str): Root directory.
        sub_dirs (list): List of subdirectories to search.
        filter_keyword (str): Keyword to filter files.
        sampling_ratio (float): Ratio of files to include.

    Returns:
        list: List of file paths.
    """
    file_list = []
    for sub_dir in sub_dirs:
        abs_dir_path = os.path.join(root_dir, sub_dir)
        files = [os.path.join(abs_dir_path, f) for f in os.listdir(abs_dir_path)]
        files = list(filter(lambda file_path: filter_keyword in file_path, files))
        files.sort()
        file_list.extend(files[:int(len(files) * sampling_ratio)])
    return file_list


def count_word_occurrences(word, root_dir, sub_dirs):
    """
    Count occurrences of a specific word across files in the target directories.

    Args:
        word (str): Word to count.
        root_dir (str): Root directory.
        sub_dirs (list): List of subdirectories to search.

    Returns:
        dict: Dictionary of word counts per directory.
    """
    word_counts = {}
    for sub_dir in sub_dirs:
        abs_dir_path = os.path.join(root_dir, sub_dir)
        files = list_files(root_dir, [sub_dir], sampling_ratio=1.0)
        word_count = 0
        for file_path in files:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                word_count += sum([len(line.split(word)) - 1 for line in lines])
        word_counts[sub_dir] = word_count
    return word_counts


def train_bpe_tokenizer(output_dir, data_dir, sub_dirs, vocab_size=32000, limit_alphabet = 2000, sampling_ratio=0.3):
    """
    Train a BPE tokenizer using target directories.

    Args:
        output_dir (str): Directory to save the tokenizer.
        data_dir (str): Root directory for training data.
        sub_dirs (list): List of subdirectories to search.
        vocab_size (int): Vocabulary size.
        sampling_ratio (float): Ratio of files to use for training.
    """
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>", fuse_unk=True))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(' '), behavior='merged_with_next'),
        pre_tokenizers.Split(Regex(r'\d|[\u2E80-\u2FDF\u3040-\u318F\u31A0-\u31BF\u31F0-\u31FF\u3400-\u4DB5\u4E00-\u9FFF\uA960-\uA97F\uAC00-\uD7FF]'), behavior='isolated'),
        pre_tokenizers.Split(Regex(r' *(\w+|[^\w\s]+)'), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.Sequence([decoders.ByteLevel()])
    trainer = BpeTrainer(
        special_tokens=["<unk>", "<s>", "</s>"],
        vocab_size=vocab_size - 256,
        limit_alphabet=limit_alphabet,
        initial_alphabet=pre_tokenizers.ByteLevel().alphabet(),
    )

    # Get training files
    training_files = list_files(data_dir, sub_dirs, sampling_ratio=sampling_ratio)
    print(f"Training on {len(training_files)} files...")

    # Train tokenizer
    tokenizer.train(files=training_files, trainer=trainer)
    save_path = os.path.join(output_dir, 'tokenizer_raw.json')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    tokenizer.save(save_path)

    with open(save_path, 'r') as f:
        x = json.load(f)
        new_vocab = {}
        new_vocab['<unk>'] = 0
        new_vocab['<s>'] = 1
        new_vocab['</s>'] = 2

        for i in range(256):
            hexn = hex(i)[2:].upper()
            s = f"<0x{hexn:>02s}>"
            new_vocab[s] = i + 3

        for k, v in x['model']['vocab'].items():
            if k not in ['<unk>', '<s>', '</s>']:
                new_vocab[k] = v + 256

        x['model']['vocab'] = new_vocab
    
    new_path = os.path.join(output_dir, "tokenizer.json")
    with open(new_path, 'w') as f:
        json.dump(x, f, indent=2, ensure_ascii=False)

    print(f"Tokenizer saved to {save_path}")


def validate_bpe_tokenizer(tokenizer_dir, tokenizer_file, sample_texts):
    """
    Validate a trained BPE tokenizer.

    Args:
        output_dir (str): Directory where the tokenizer is saved.
        tokenizer_dir (str): Subdirectory of the tokenizer.
        tokenizer_file (str): Tokenizer file name.
        sample_texts (list): List of texts to tokenize and decode.
    """
    tokenizer_path = os.path.join(tokenizer_dir, tokenizer_file)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        eos_token='</s>',
    )
    print(f"Validating tokenizer from {tokenizer_path}")

    # Tokenize and decode sample texts
    for text in sample_texts:
        tokenized_output = tokenizer([text], max_length=10, truncation=False, return_overflowing_tokens=True)
        print(f"Original: {text}")
        print(f"Tokenized: {tokenized_output}")


if __name__ == "__main__":
    # Configuration
    output_dir = "./tokenizer"
    tokenizer_dir = "./tokenizer"
    tokenizer_file = "tokenizer.json"
    data_dir = "./datasets"
    sub_dirs = ['english_wikipedia', 'english_gutenberg', 'english_books3', 'english_c4']

    # Step 0: Count word occurrences (to checkout if overflow)
    # word_occurrences = count_word_occurrences('the', data_dir, sub_dirs)
    # for directory, count in word_occurrences.items():
    #     print(f"{directory}: {count} occurrences of 'the'")

    # Step 1: Train BPE tokenizer
    train_bpe_tokenizer(output_dir, data_dir, sub_dirs, vocab_size=32000, sampling_ratio=1.0)

    # Step 2: Validate BPE tokenizer
    test_texts = [
        "Let's train a 1 billion Large Lanugage Model!",
        "This is the first step of training a tokenizer."
    ]
    validate_bpe_tokenizer(tokenizer_dir, tokenizer_file, test_texts)
