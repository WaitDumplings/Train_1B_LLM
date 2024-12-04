import os
import time
import copy
import json
import torch
import random
import pickle
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

# Function to generate attention masks for token sequences
def mask_generator(x):
    # Create a square zero matrix of shape [x.shape[0], x.shape[0]]
    mask = np.zeros([x.shape[0], x.shape[0]], dtype=np.float32)
    # Identify indices where the token equals 2
    indices = np.where(x == 2)[0]
    for i in indices:
        if i + 1 == x.shape[0]:  # Skip if it's the last index
            continue
        # Set mask values to -1 for elements beyond current index
        mask[i+1 :, 0 : i+1] = -1
    return mask

# Custom dataset class for GPT training
class GPTDataset(Dataset):
    def __init__(self, data_root, data_dirs, sequence_length, set_name, ratio=1.0, shuffle=True, mask_flag=False):
        random.seed(1)  # Set random seed for reproducibility
        data_paths = []  # Store file paths
        splitters = []   # Store data split delimiters

        # Iterate over provided data directories and their configurations
        for data_dir, es in data_dirs.items():
            data_epochs, data_splitter = es
            files = self.get_all_files(data_root, data_dir, set_name, ratio)  # Fetch files
            if set_name == "validation":  # Use single epoch for validation
                data_epochs = 1
            splitters += [data_splitter] * data_epochs  # Add splitters for each epoch

            tmp_paths = [[] for _ in range(data_epochs)]  # Temporary file storage for shuffling
            knt = 0
            for _ in range(data_epochs):
                fc = copy.copy(files)  # Copy files for shuffling
                if shuffle:
                    random.shuffle(fc)  # Shuffle file order
                for f in fc:
                    tmp_paths[knt % data_epochs].append(f)
                    knt += 1
            data_paths += tmp_paths  # Append shuffled paths

        # Initialize dataset attributes
        self.tokens = np.empty((1, sequence_length))
        self.labels = np.empty((1, sequence_length))
        self.files = list(zip(*data_paths))
        self.splitters = splitters
        self.sequence_length = sequence_length
        self.mask_flag = mask_flag

    # Static method to retrieve all relevant files from a directory
    @staticmethod
    def get_all_files(root, data_dir, set_name='train', ratio=1.0):
        data_path = os.path.join(root, data_dir)  # Build directory path
        files = list(map(lambda x: os.path.join(data_path, x), os.listdir(data_path)))  # List all files
        files = list(filter(lambda x: set_name in x, files))  # Filter by set name
        files.sort()  # Sort file paths
        return files[:int(len(files) * ratio)]  # Limit by ratio

    # Tokenization function for single data point or batch
    def tokenize(self, index, tokenizer, token_dump_path, is_batch=False):
        if index >= len(self.files):  # Ensure index is valid
            return
        
        eos = tokenizer.encode('</s>')[0]  # Encode end-of-sequence token
        data = []  # Initialize data storage

        # Read data files and split by defined delimiters
        for file, splitter in zip(self.files[index], self.splitters):
            with open(file, 'r') as f:
                dt = f.readlines()
                dt = ''.join(dt).split(splitter)
                dt = list(map(lambda x: x.strip(), dt))
                data += dt

        # Truncate long texts to improve processing efficiency
        chunk = 20000
        data = list(map(lambda x: x[:chunk], data))
        random.shuffle(data)  # Shuffle the data

        # Tokenize the data
        if is_batch:
            tokens = tokenizer(data)['input_ids']
        else:
            step = 100
            tokens = []
            for i in range(0, len(data), step):
                tokens.extend(tokenizer(data[i:i+step])['input_ids'])

        # Add padding with end-of-sequence tokens
        tokens = list(map(lambda x: x + [eos] * 10, tokens))
        tokens = [elem for tl in tokens for elem in tl]  # Flatten token lists

        # Truncate tokens to fit into the sequence length
        truncate_len = len(tokens) // self.sequence_length * self.sequence_length
        tokens = tokens[:truncate_len]
        tokens = np.array(tokens, dtype=np.int32).reshape((-1, self.sequence_length))

        labels = tokens.copy()  # Create labels identical to input tokens
        pickle.dump((len(data), tokens, labels), open(token_dump_path, 'wb'))  # Save tokens and labels
    # Tokenization function with support for input and target packing
    def tokenize_with_packing(self, index, tokenizer, token_dump_path, is_batch=False):
        if index >= len(self.files):  # Ensure index is valid
            return
        
        eos = tokenizer.encode('</s>')[0]  # Encode end-of-sequence token
        IGNORE_INDEX = -100  # Default value for ignored positions in CrossEntropyLoss
        data = []  # Store input sequences
        target = []  # Store target sequences

        # Process each file to extract inputs and targets
        for file in self.files[index]:
            with open(file, 'r') as f:
                dt = f.readlines()
                da = ""  # Temporary variable to store input text
                for i in range(0, len(dt)):
                    if dt[i].startswith("Input: "):  # Identify input lines
                        da = dt[i][7:].strip() + '\n'  # Remove "Input: " prefix
                    elif dt[i].startswith("Response: "):  # Identify response lines
                        if len(da) == 0:  # Skip if no input is recorded
                            continue
                        data.append(da)  # Append input
                        target.append(dt[i].strip())  # Append corresponding response
                        da = ""  # Reset input text
                    elif dt[i] == "_\n":  # Skip special marker lines
                        continue
                    else:
                        da += dt[i].strip() + "\n"  # Continue appending text

        # Tokenize input and target sequences
        if is_batch:
            tokens = tokenizer(data)['input_ids']
            label_tokens = tokenizer(target)['input_ids']
        else:
            step = 100
            tokens, label_tokens = [], []
            for i in range(0, len(data), step):
                tokens.extend(tokenizer(data[i:i+step])['input_ids'])
                label_tokens.extend(tokenizer(target[i:i+step])['input_ids'])

        # Add end-of-sequence tokens to label sequences
        label_tokens = list(map(lambda x: x + [eos], label_tokens))
        
        # Combine tokens and label tokens for packed inputs and labels
        inputs = list(map(lambda x: x[0] + x[1], zip(tokens, label_tokens)))
        labels = list(map(lambda x: [IGNORE_INDEX] * len(x[0]) + x[1], zip(tokens, label_tokens)))

        # Shuffle inputs and labels in unison
        zip_inputs_labels = list(zip(inputs, labels))
        random.shuffle(zip_inputs_labels)
        inputs, labels = zip(*zip_inputs_labels)

        # Prepare for batching by padding or truncating sequences
        token_lists = []
        padding_labels = []
        cur_tok = inputs[0]  # Initialize with the first token
        cur_lab = labels[0]  # Initialize with the first label
        for i, (ts, lbs) in enumerate(zip(inputs[1:], labels[1:])):
            if len(cur_tok) + len(ts) <= self.sequence_length:
                cur_tok += ts  # Append tokens if within sequence length
                cur_lab += lbs  # Append labels
            else:
                if len(cur_tok) <= self.sequence_length:
                    # Pad current tokens and labels
                    pad_num = self.sequence_length - len(cur_tok)
                    cur_tok += [0] * pad_num
                    cur_lab += [IGNORE_INDEX] * pad_num
                else:
                    # Truncate to fit sequence length
                    cur_tok = cur_tok[-self.sequence_length:]
                    cur_lab = cur_lab[-self.sequence_length:]
                token_lists.append(cur_tok)  # Add to batch
                padding_labels.append(cur_lab)
                cur_tok = ts
                cur_lab = lbs

        # Convert tokens and labels to numpy arrays
        token_lists = np.array(token_lists, dtype=np.int32)
        padding_labels = np.array(padding_labels, dtype=np.int32)

        # Save processed data as a pickle file
        pickle.dump((len(data), token_lists, padding_labels), open(token_dump_path, 'wb'))

    # Load tokenized samples from a file
    def load_samples(self, token_dump_path):
        num_origin_samples, self.tokens, self.labels = pickle.load(open(token_dump_path, 'rb'))
        return num_origin_samples

    # Reset the dataset's token and label arrays
    def reset_samples(self):
        self.tokens = np.empty((1, self.sequence_length))
        self.labels = np.empty((1, self.sequence_length))

    # Get the total number of samples in the dataset
    def __len__(self):
        return self.tokens.shape[0]

    # Retrieve a single sample or sample with attention masks
    def __getitem__(self, index):
        if self.mask_flag:
            att_masks = mask_generator(self.tokens[index])  # Generate attention mask
            return self.tokens[index], self.labels[index], att_masks
        else:
            return self.tokens[index], self.labels[index]

# Custom random sampler for creating batches
class RandSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, shuffle=True):
        self.batch_size = batch_size
        self.order = list(range(len(data_source)))  # Create an ordered index list
        self.total_size = len(self.order) - len(self.order) % self.batch_size
        if not drop_last:
            self.total_size += batch_size

        if shuffle:  # Shuffle the order if required
            random.shuffle(self.order)
        self.groups = []  # Group indices into batches
        for i in range(0, self.total_size, self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    # Yield batches one by one
    def __iter__(self):
        for group in self.groups:
            yield group

    # Return the total number of groups (batches)
    def __len__(self):
        return len(self.groups)

# Distributed random sampler for parallelized training
class DistRandSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, shuffle=True):
        self.rank = dist.get_rank()  # Get the process rank
        self.num_replicas = dist.get_world_size()  # Get total number of processes
        self.batch_size = batch_size
        self.order = list(range(len(data_source)))  # Create an ordered index list
        self.total_size = len(self.order) - len(self.order) % (self.num_replicas * self.batch_size)
        if not drop_last:
            self.total_size += self.num_replicas * self.batch_size

        if shuffle:  # Shuffle the order if required
            g = torch.Generator()
            g.manual_seed(-1)  # Set a fixed seed for reproducibility
            self.order = torch.randperm(len(self.order), generator=g).tolist()
        self.groups = []  # Group indices for distributed sampling
        for i in range(self.rank * self.batch_size, self.total_size, self.num_replicas * self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    # Yield batches for the specific process
    def __iter__(self):
        for group in self.groups:
            yield group

    # Return the total number of groups (batches)
    def __len__(self):
        return len(self.groups)

# Collate function to prepare input and labels for training
def FixCollater(batch):
    inputs, labels = zip(*batch)  # Unpack inputs and labels
    inputs = torch.from_numpy(np.array(inputs)).to(torch.long)  # Convert to tensor
    labels = torch.from_numpy(np.array(labels)).to(torch.long)  # Convert to tensor
    return inputs, labels

# Collate function to prepare input, labels, and attention masks for training
def MaskCollater(batch):
    inputs, labels, att_masks = zip(*batch)  # Unpack inputs, labels, and masks
    inputs = torch.from_numpy(np.array(inputs)).to(torch.long)  # Convert inputs to tensor
    labels = torch.from_numpy(np.array(labels)).to(torch.long)  # Convert labels to tensor
    att_masks = torch.from_numpy(np.array(att_masks)).to(torch.float32)  # Convert masks to tensor
    return inputs, labels, att_masks
