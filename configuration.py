from dataclasses import dataclass

@dataclass
class RetrieverConfig_tiny:
    gpu_num = 1
    batch_size = 4
    gradient_accumulation_steps = 3
    sequence_length = 1024
    learning_rate = 6e-4
    min_lr = 6e-5
    vocab_size = 32000
    num_layers = 1
    hidden_size = 512
    kv_num_heads = 1
    query_num_heads = 2
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-1
    warmup_iters = 20
    max_iters = 160000
    lr_decay_iters = 160000
    grad_clip = 1.0

@dataclass
class RetrieverConfig_small:
    gpu_num = 4
    batch_size = 32
    gradient_accumulation_steps = 3
    sequence_length = 1024
    learning_rate = 6e-4
    min_lr = 6e-5
    vocab_size = 32000
    num_layers = 6
    hidden_size = 512
    kv_num_heads = 1
    query_num_heads = 8
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-1
    warmup_iters = 2000
    max_iters = 160000
    lr_decay_iters = 160000
    grad_clip = 1.0

@dataclass
class RetrieverConfig_medium:
    gpu_num = 4
    batch_size = 8
    gradient_accumulation_steps = 20
    sequence_length = 1024
    learning_rate = 6e-4
    min_lr = 6e-5
    vocab_size = 32000
    num_layers = 18
    hidden_size = 1200
    kv_num_heads = 1
    query_num_heads = 16
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-1
    warmup_iters = 2000
    max_iters = 200000
    lr_decay_iters = 200000
    grad_clip = 1.0

@dataclass
class RetrieverConfig_medium_finetune:
    batch_size = 16
    gradient_accumulation_steps = 8
    learning_rate = 1e-4
    min_lr = 1e-5
    max_iters = 20000
    lr_decay_iters = 20000

@dataclass
class RetrieverConfig_large:
    gpu_num = 4
    batch_size = 8
    gradient_accumulation_steps = 20
    sequence_length = 1024
    learning_rate = 6e-4
    min_lr = 6e-5
    vocab_size = 32000
    num_layers = 24
    hidden_size = 1280
    kv_num_heads = 1
    query_num_heads = 16
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-1
    warmup_iters = 2000
    max_iters = 200000
    lr_decay_iters = 200000
    grad_clip = 1.0