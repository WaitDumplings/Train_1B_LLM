import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import torch
import gradio as gr
from functools import partial
from model import retriever
from configuration import *
from transformers import PreTrainedTokenizerFast

def response_func(message, history, model, tokenizer):
    """
    Generate a chatbot response based on the input message and conversation history.
    
    Parameters:
    - message: The latest user query/message.
    - history: A list of (question, answer) tuples representing the conversation history.
    - model: The pretrained model used to generate responses.
    - tokenizer: Tokenizer to preprocess the input and decode the output.
    
    Returns:
    - response: The chatbot's response to the input message.
    """
    begin_time = time.time()  # Record the start time of the function execution
    query_length = len(message)  # Initialize query length with the length of the latest message
    for q, a in history:  # Loop through the conversation history
        query_length += len(q) + len(a)  # Add the length of previous queries and answers
    print(message, history)  # Print the message and history for debugging
    response = model.chat(tokenizer, message, history, temperature=1.0, top_k=1)  # Generate a response
    print(f"query_length: {query_length}, response_length: {len(response)}, consume {time.time() - begin_time:.1f}s")
    return response

# Define the main function to initialize and run the chatbot application
def main(gpu, arch, dtype, model_root, model_path, tokenizer_path):
    """
    Main function to set up the model, tokenizer, and Gradio interface for the chatbot.
    
    Parameters:
    - gpu: The GPU device to be used for computations.
    - arch: The model architecture to be used.
    - dtype: The data type for model computations (e.g., float16).
    - model_root: Root directory where model and tokenizer files are located.
    - model_path: Relative path to the model checkpoint file.
    - tokenizer_path: Relative path to the tokenizer JSON file.
    """
    # Map string representations of data types to PyTorch data types
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    # Load the tokenizer from the specified path
    tokenizer_path = os.path.join(model_root, tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    # Set the GPU device
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    # Load the model from the specified path
    model_path = os.path.join(model_root, model_path)
    model = arch(device, ptdtype, config, pretrained=True, model_path=model_path, flash=True)  # Initialize the model
    model.cuda(gpu)  # Move the model to the GPU
    # model = torch.compile(model)
    model = model.eval()  # Set the model to evaluation mode

    # Set up the Gradio chat interface
    demo = gr.ChatInterface(partial(response_func, model=model, tokenizer=tokenizer), title="CS 228 Chat Bot")
    demo.launch(server_port=7870, share=False) # shape = True if you wanna other pepole can visit this link.

if __name__ == "__main__":
    arch = retriever
    config = MHA_config()
    dtype = "float16"
    model_root = "."
    model_path = "ckpt/FT_LLM_MHA.pth.tar"
    tokenizer_path = "tokenizer/tokenizer_new.json"
    main(0, arch, dtype, model_root, model_path, tokenizer_path)