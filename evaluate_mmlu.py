import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time
import torch
import random
import pandas as pd
import numpy as np
from model import retriever
from tokenizers import Tokenizer
from torch.cuda.amp import autocast
from evaluate_datasets import *
from configuration import *
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

def mmlu(model, tokenizer, data_root, config, device, k_shot=0):
    files_list = list(filter(lambda x: x.endswith("txt"), os.listdir(data_root)))
    choices = ["A", "B", "C", "D"]
    choices_token = [tokenizer.encode(i)[0] for i in choices]
    target = ""
    Accuracy = 0
    count = 0

    for file in tqdm(files_list):
        file_path = os.path.join(data_root, file)
        question = ""
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("question: "):
                    question = line
                    target = ""
                elif line.startswith("choices: "):
                    question += line
                elif line.startswith("answer: "):
                    target = line[-2]
                    count += 1
                    question += "answer: "

                    prompt = f"The following are multiple-choice questions, please choose the correct answer.\n {question}"

                    inputs = tokenizer.encode(prompt)
                    inputs = torch.from_numpy(np.array([inputs])).to(torch.long)
                    if inputs.shape[1] > config.sequence_length:
                        inputs = inputs[:, -config.sequence_length:]
                    inputs = inputs.to(device)

                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(inputs)
                        logits = logits[:, -1].cpu()
                        valid_probs = logits[0][choices_token]
                        next_token = torch.argmax(valid_probs).item()
                        Accuracy += (target == tokenizer.decode(choices_token[next_token]))

    print("MMLU Accuracy is {}".Accuracy/count)

def main(arch, config, model_root, model_path, tokenizer_path, data_root):
    tokenizer_path = os.path.join(model_root, tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    device = torch.device('cuda')
    model = None
    model_path = os.path.join(model_root, model_path)
    model = arch(device, torch.bfloat16, config, pretrained=True, model_path=model_path, flash=True)
    model.cuda()
    # model = torch.compile(model)
    model = model.eval()

    with torch.no_grad():
        mmlu(model, tokenizer, data_root, config, device, k_shot=0)

if __name__ == "__main__":
    arch = retriever
    config = RetrieverConfig_medium_MQA()
    model_root = "."
    model_path = "ckpt/LLM_MQA.pth.tar"
    tokenizer_path = "tokenizer/tokenizer_v2_600G.json"
    data_root = "evaluate_datasets/data_txt/evaluate_en_mmlu"

    main(arch, config, model_root, model_path, tokenizer_path, data_root)