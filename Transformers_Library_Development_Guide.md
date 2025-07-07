# Transformers Library Development Guide

## Creating Highly Efficient, Parallelizable, and Maintainable Code for Fast Inference and Training

This comprehensive guide provides best practices for developing efficient transformer models using the Hugging Face ecosystem, with insights drawn from the TinyStories methodology and modern optimization techniques.

## Table of Contents

1. [Architecture Design Principles](#architecture-design-principles)
2. [Efficient Model Implementation](#efficient-model-implementation)
3. [Multi-GPU Training Strategies](#multi-gpu-training-strategies)
4. [Optimization Techniques](#optimization-techniques)
5. [Memory Management](#memory-management)
6. [Caching and KV Cache Strategies](#caching-and-kv-cache-strategies)
7. [Batch Processing and Data Loading](#batch-processing-and-data-loading)
8. [Code Organization and Modularity](#code-organization-and-modularity)
9. [Performance Monitoring and Debugging](#performance-monitoring-and-debugging)
10. [Production Deployment](#production-deployment)

## Architecture Design Principles

### 1. Modular Design
- **Separate concerns**: Keep data processing, model architecture, training logic, and evaluation separate
- **Configurable components**: Use configuration files for hyperparameters, model architecture, and training settings
- **Reusable modules**: Design components that can be easily swapped or extended

```python
# Example: Modular configuration approach
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int = 10000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    max_position_embeddings: int = 512
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 5e-4
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
```

### 2. Scalable Architecture
- **Parameter-efficient designs**: Follow TinyStories approach for small but effective models
- **Layer-wise scaling**: Design models that can scale from 1M to 100M+ parameters
- **Attention mechanisms**: Implement efficient attention patterns (local, sparse, or sliding window)

## Efficient Model Implementation

### 1. Core Model Components

```python
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from torch.nn import functional as F

class EfficientTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = EfficientAttention(config)
        self.mlp = EfficientMLP(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attention_mask=None, past_key_values=None, use_cache=False):
        # Pre-norm architecture for better training stability
        attn_output, present_key_value = self.attention(
            self.ln1(x), 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        
        outputs = (x,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
```

### 2. Efficient Attention Implementation

```python
class EfficientAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Use single linear layer for efficiency
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, attention_mask=None, past_key_values=None, use_cache=False):
        B, T, C = x.size()
        
        # Compute Q, K, V in one go
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key values for caching
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        present_key_value = (k, v) if use_cache else None
        
        # Use SDPA for efficiency (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Fallback implementation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn_weights += attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, present_key_value
```

## Multi-GPU Training Strategies

### 1. Choosing the Right Parallelism Strategy

Based on your setup and model size:

| Setup | Model Fits Single GPU | Strategy |
|-------|----------------------|----------|
| Single Node/Multi-GPU | Yes | DistributedDataParallel (DDP) |
| Single Node/Multi-GPU | No | Pipeline Parallel + ZeRO |
| Multi-Node/Multi-GPU | Any | ZeRO-3 or 3D Parallelism |

### 2. Implementation with Accelerate

```python
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
from torch.utils.data import DataLoader

def setup_training():
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision='bf16',  # Use bf16 for better stability
        log_with="wandb",
        project_dir="./logs"
    )
    
    # Set seed for reproducibility
    set_seed(42)
    
    return accelerator

def train_with_accelerate(model, train_dataloader, optimizer, accelerator):
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
            if batch_idx % 100 == 0:
                accelerator.log({"loss": loss.item()}, step=batch_idx)
```

### 3. DeepSpeed Integration

```python
# deepspeed_config.json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-4,
            "warmup_num_steps": 1000
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

## Optimization Techniques

### 1. Mixed Precision Training

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-4,
    
    # Mixed precision settings
    bf16=True,  # Preferred over fp16 for stability
    dataloader_pin_memory=True,
    
    # Optimization settings
    optim="adamw_torch_fused",  # Faster AdamW implementation
    gradient_checkpointing=True,  # Save memory at cost of speed
    
    # Compilation (PyTorch 2.0+)
    torch_compile=True,
    torch_compile_backend="inductor",
    
    # Memory management
    torch_empty_cache_steps=50,
    
    # Data loading optimization
    dataloader_num_workers=4,
    remove_unused_columns=False,
)
```

### 2. Gradient Accumulation and Checkpointing

```python
class OptimizedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Use automatic mixed precision
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            
        # Scale loss for gradient accumulation
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        # Backward pass with gradient scaling
        self.accelerator.backward(loss)
        
        return loss.detach()
```

## Memory Management

### 1. Efficient Memory Usage Patterns

```python
def optimize_memory_usage():
    # Clear cache periodically
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Use gradient checkpointing for large models
    model.gradient_checkpointing_enable()
    
    # Optimize data types
    model = model.to(dtype=torch.bfloat16)  # Use bf16 for better range
    
    # Use memory-efficient optimizers
    from bitsandbytes.optim import AdamW8bit
    optimizer = AdamW8bit(model.parameters(), lr=5e-4)
```

### 2. Dynamic Batch Sizing

```python
class DynamicBatchSampler:
    def __init__(self, dataset, max_tokens=4096, max_batch_size=32):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
    
    def __iter__(self):
        batch = []
        current_tokens = 0
        
        for idx, item in enumerate(self.dataset):
            item_tokens = len(item['input_ids'])
            
            if (current_tokens + item_tokens > self.max_tokens or 
                len(batch) >= self.max_batch_size) and batch:
                yield batch
                batch = []
                current_tokens = 0
            
            batch.append(idx)
            current_tokens += item_tokens
        
        if batch:
            yield batch
```

## Caching and KV Cache Strategies

### 1. Implementing Efficient KV Cache

```python
from transformers import Cache, DynamicCache

class OptimizedGenerationMixin:
    def generate_with_cache(self, input_ids, max_new_tokens=50, use_cache=True):
        if use_cache:
            past_key_values = DynamicCache()
        else:
            past_key_values = None
            
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Only process the last token if using cache
            if use_cache and past_key_values.get_seq_length() > 0:
                input_for_forward = generated_ids[:, -1:]
            else:
                input_for_forward = generated_ids
                
            outputs = self(
                input_ids=input_for_forward,
                past_key_values=past_key_values,
                use_cache=use_cache
            )
            
            # Sample next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.multinomial(
                F.softmax(next_token_logits, dim=-1), 
                num_samples=1
            )
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if use_cache:
                past_key_values = outputs.past_key_values
                
        return generated_ids
```

### 2. Memory-Efficient Cache Management

```python
class MemoryEfficientCache(Cache):
    def __init__(self, max_cache_length=1024):
        super().__init__()
        self.max_cache_length = max_cache_length
        self.key_cache = []
        self.value_cache = []
        
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Concatenate with existing cache
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
            
            # Trim cache if too long
            if self.key_cache[layer_idx].size(-2) > self.max_cache_length:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, -self.max_cache_length:, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, -self.max_cache_length:, :]
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

## Batch Processing and Data Loading

### 1. Efficient Data Pipeline

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np

class EfficientTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, cache_tokenization=True):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

        if cache_tokenization:
            self._tokenize_all()

    def _tokenize_all(self):
        """Pre-tokenize all texts for faster training"""
        self.tokenized_texts = []
        for text in self.texts:
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )
            self.tokenized_texts.append(tokens)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.cache_tokenization:
            return self.tokenized_texts[idx]
        else:
            return self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )

def create_efficient_dataloader(dataset, batch_size=32, num_workers=4):
    """Create optimized DataLoader with proper collation"""

    def collate_fn(batch):
        # Efficient batching with padding
        input_ids = [item['input_ids'].squeeze(0) for item in batch]
        attention_masks = [item['attention_mask'].squeeze(0) for item in batch]

        # Pad to max length in batch (not global max)
        max_len = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(input_ids, attention_masks):
            pad_length = max_len - len(ids)
            padded_input_ids.append(
                torch.cat([ids, torch.zeros(pad_length, dtype=ids.dtype)])
            )
            padded_attention_masks.append(
                torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])
            )

        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
            'labels': torch.stack(padded_input_ids)  # For causal LM
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )
```

### 2. Smart Batching Strategies

```python
class SmartBatchSampler:
    """Groups samples by length for efficient batching"""

    def __init__(self, dataset, batch_size, max_tokens=None, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.drop_last = drop_last

        # Sort indices by sequence length
        self.length_sorted_indices = self._sort_by_length()

    def _sort_by_length(self):
        lengths = []
        for i, item in enumerate(self.dataset):
            if hasattr(item, 'input_ids'):
                lengths.append((len(item.input_ids), i))
            else:
                # Fallback: estimate length
                lengths.append((len(str(item)), i))

        # Sort by length
        lengths.sort(key=lambda x: x[0])
        return [idx for _, idx in lengths]

    def __iter__(self):
        batch = []
        current_max_len = 0

        for idx in self.length_sorted_indices:
            item_len = len(self.dataset[idx].get('input_ids', []))

            # Check if adding this item would exceed limits
            new_max_len = max(current_max_len, item_len)
            total_tokens = new_max_len * (len(batch) + 1)

            if (len(batch) >= self.batch_size or
                (self.max_tokens and total_tokens > self.max_tokens)) and batch:
                yield batch
                batch = []
                current_max_len = 0

            batch.append(idx)
            current_max_len = max(current_max_len, item_len)

        if batch and not self.drop_last:
            yield batch
```

## Code Organization and Modularity

### 1. Project Structure

```
project/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── configuration.py
│   │   ├── modeling.py
│   │   └── tokenization.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── data_collator.py
│   │   └── callbacks.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   └── optimization.py
│   └── data/
│       ├── __init__.py
│       ├── dataset.py
│       └── preprocessing.py
├── configs/
│   ├── model_configs/
│   ├── training_configs/
│   └── data_configs/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
├── tests/
└── requirements.txt
```

### 2. Configuration Management

```python
# configs/base_config.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml
import json

@dataclass
class BaseConfig:
    """Base configuration class with serialization support"""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str):
        config_dict = self.to_dict()
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)

    @classmethod
    def from_file(cls, path: str):
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                config_dict = json.load(f)
        return cls(**config_dict)

@dataclass
class ModelConfig(BaseConfig):
    # Architecture
    vocab_size: int = 10000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    intermediate_size: Optional[int] = None
    max_position_embeddings: int = 512

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Activation
    activation_function: str = "gelu"

    # Initialization
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

@dataclass
class TrainingConfig(BaseConfig):
    # Basic training settings
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1

    # Optimization
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1

    # Mixed precision and optimization
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    torch_compile: bool = False

    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Logging and saving
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 3

    # Evaluation
    evaluation_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
```

### 3. Modular Training Pipeline

```python
# src/training/trainer.py
from transformers import Trainer, TrainingArguments
from typing import Dict, Optional, Any
import torch
import wandb

class ModularTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        custom_loss_fn=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs
        )
        self.custom_loss_fn = custom_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation with support for different loss functions"""
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if self.custom_loss_fn is not None:
            loss = self.custom_loss_fn(outputs, labels)
        else:
            # Default loss computation
            if labels is not None:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, labels)
                else:
                    loss = outputs.loss
            else:
                loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging with custom metrics"""
        # Add custom metrics
        if hasattr(self.model, 'get_memory_usage'):
            logs['memory_usage'] = self.model.get_memory_usage()

        if torch.cuda.is_available():
            logs['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
            logs['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3

        super().log(logs)

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log(logs, step=self.state.global_step)

## Performance Monitoring and Debugging

### 1. Performance Profiling

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
from contextlib import contextmanager

class PerformanceMonitor:
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_times = []
        self.memory_usage = []

    @contextmanager
    def profile_step(self, step_name="training_step"):
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            with record_function(step_name):
                yield
        finally:
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            step_time = end_time - start_time
            memory_delta = end_memory - start_memory

            self.step_times.append(step_time)
            self.memory_usage.append(memory_delta)

            if len(self.step_times) % self.log_interval == 0:
                self.log_stats()

    def log_stats(self):
        if self.step_times:
            avg_time = sum(self.step_times[-self.log_interval:]) / min(len(self.step_times), self.log_interval)
            avg_memory = sum(self.memory_usage[-self.log_interval:]) / min(len(self.memory_usage), self.log_interval)

            print(f"Avg step time: {avg_time:.4f}s, Avg memory delta: {avg_memory/1024**2:.2f}MB")

def profile_model_training(model, dataloader, num_steps=10):
    """Profile model training for performance analysis"""

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        model.train()
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            with record_function("forward_pass"):
                outputs = model(**batch)
                loss = outputs.loss

            with record_function("backward_pass"):
                loss.backward()

            with record_function("optimizer_step"):
                # optimizer.step() would go here
                pass

    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Save detailed trace
    prof.export_chrome_trace("trace.json")

    return prof
```

### 2. Memory Debugging

```python
class MemoryTracker:
    def __init__(self):
        self.snapshots = []

    def take_snapshot(self, name=""):
        if torch.cuda.is_available():
            snapshot = {
                'name': name,
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
                'max_reserved': torch.cuda.max_memory_reserved()
            }
            self.snapshots.append(snapshot)
            return snapshot
        return None

    def print_memory_summary(self):
        if not torch.cuda.is_available():
            print("CUDA not available")
            return

        print("\n=== Memory Usage Summary ===")
        for i, snapshot in enumerate(self.snapshots):
            print(f"{i}: {snapshot['name']}")
            print(f"  Allocated: {snapshot['allocated']/1024**3:.2f} GB")
            print(f"  Reserved: {snapshot['reserved']/1024**3:.2f} GB")
            print(f"  Max Allocated: {snapshot['max_allocated']/1024**3:.2f} GB")
            print(f"  Max Reserved: {snapshot['max_reserved']/1024**3:.2f} GB")
            print()

    def clear_snapshots(self):
        self.snapshots.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

# Usage example
memory_tracker = MemoryTracker()

# During training
memory_tracker.take_snapshot("before_model_load")
model = load_model()
memory_tracker.take_snapshot("after_model_load")

# ... training code ...

memory_tracker.take_snapshot("after_training")
memory_tracker.print_memory_summary()
```

### 3. Gradient Monitoring

```python
class GradientMonitor:
    def __init__(self, model):
        self.model = model
        self.gradient_norms = []
        self.parameter_norms = []

    def compute_gradient_norm(self):
        total_norm = 0.0
        param_count = 0

        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        return total_norm

    def compute_parameter_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = total_norm ** (1. / 2)
        self.parameter_norms.append(total_norm)
        return total_norm

    def check_gradient_health(self, threshold=10.0):
        """Check for gradient explosion or vanishing"""
        if self.gradient_norms:
            recent_norm = self.gradient_norms[-1]
            if recent_norm > threshold:
                print(f"Warning: Large gradient norm detected: {recent_norm:.4f}")
            elif recent_norm < 1e-7:
                print(f"Warning: Very small gradient norm detected: {recent_norm:.4e}")
```

## Production Deployment

### 1. Model Optimization for Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.jit import script, trace
import onnx
import onnxruntime as ort

class ModelOptimizer:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def optimize_for_inference(self):
        """Apply various optimizations for inference"""
        # Set to evaluation mode
        self.model.eval()

        # Enable optimizations
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = True

        # Compile with torch.compile (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # Use SDPA for attention
        if hasattr(self.model.config, 'attn_implementation'):
            self.model.config.attn_implementation = "sdpa"

        return self.model

    def export_to_onnx(self, output_path, sample_input_ids=None):
        """Export model to ONNX format"""
        if sample_input_ids is None:
            sample_input_ids = torch.randint(0, 1000, (1, 10))

        # Export to ONNX
        torch.onnx.export(
            self.model,
            sample_input_ids,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size', 1: 'sequence'}
            }
        )

        return output_path

    def quantize_model(self, quantization_type="dynamic"):
        """Apply quantization for smaller model size"""
        if quantization_type == "dynamic":
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        elif quantization_type == "static":
            # Static quantization requires calibration data
            # This is a simplified example
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self.model, inplace=True)
            # ... calibration step would go here ...
            quantized_model = torch.quantization.convert(self.model, inplace=False)

        return quantized_model
```

### 2. Serving Infrastructure

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import asyncio
from typing import List, Optional
import uvicorn

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    generation_time: float

class ModelServer:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

        # Optimize for inference
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.time()

        # Tokenize input
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        generation_time = time.time() - start_time

        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            generation_time=generation_time
        )

# FastAPI app
app = FastAPI(title="Transformer Model Server")
model_server = None

@app.on_event("startup")
async def startup_event():
    global model_server
    model_server = ModelServer("path/to/your/model")

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        response = await model_server.generate(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_server is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Best Practices Summary

### 1. Development Workflow
- **Start small**: Begin with TinyStories-style small models for rapid iteration
- **Profile early**: Use profiling tools to identify bottlenecks before scaling
- **Modular design**: Keep components separate and configurable
- **Version control**: Track model configurations, training scripts, and data preprocessing

### 2. Training Optimization
- **Mixed precision**: Use bf16 for better stability than fp16
- **Gradient accumulation**: Balance memory usage and effective batch size
- **Smart batching**: Group samples by length to minimize padding
- **Caching**: Implement efficient KV caching for generation tasks

### 3. Multi-GPU Strategy
- **Single node**: Use DDP for models that fit on one GPU
- **Large models**: Use ZeRO-3 or pipeline parallelism
- **Multi-node**: Combine data, pipeline, and tensor parallelism (3D parallelism)

### 4. Memory Management
- **Monitor usage**: Track memory consumption throughout training
- **Gradient checkpointing**: Trade compute for memory when needed
- **Dynamic batching**: Adjust batch sizes based on sequence lengths
- **Cache management**: Implement memory-efficient caching strategies

### 5. Production Deployment
- **Model optimization**: Use torch.compile, quantization, and ONNX export
- **Serving infrastructure**: Implement async serving with proper error handling
- **Monitoring**: Track inference latency, throughput, and resource usage
- **Scaling**: Use load balancing and auto-scaling for production workloads

## Conclusion

This guide provides a comprehensive framework for developing efficient, scalable transformer models. The key principles are:

1. **Modularity**: Design reusable, configurable components
2. **Efficiency**: Optimize for both training and inference performance
3. **Scalability**: Plan for multi-GPU and distributed training from the start
4. **Maintainability**: Use proper code organization and monitoring
5. **Production-readiness**: Consider deployment requirements early in development

By following these practices and leveraging the Hugging Face ecosystem, you can build transformer models that are both powerful and practical for real-world applications.

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
```
```
