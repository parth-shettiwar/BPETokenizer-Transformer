from collections.abc import Iterable, Iterator
import json
import regex as re
# multiprocessing
import multiprocessing
import pathlib
import torch
from tqdm import tqdm
import math 
from .transformer import *
import math
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import numpy as np
# wandb
import wandb
from .bpe import Tokenizer
from .transformer import TransformerLM
import time


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_dim = logits.size(-1)
    logits = logits.view(-1, vocab_dim)
    targets = targets.view(-1)
    
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)
    sum_exp_logits = torch.sum(exp_logits, dim=-1, keepdim=True)
    log_probs = logits - max_logits - torch.log(sum_exp_logits)
    
    targets_flat = targets.view(-1)
    log_probs_flat = log_probs.view(-1, vocab_dim)
    loss = -torch.sum(log_probs_flat[torch.arange(log_probs_flat.size(0)), targets_flat]) / log_probs_flat.size(0)
    return loss

def learning_rate_schedule(t, alpha_max, alpha_min, warmup_iters, cosine_cycle_iters):
    if t<warmup_iters:
        return alpha_max * (t / warmup_iters)
    elif t<=cosine_cycle_iters:
        return alpha_min + (alpha_max - alpha_min) * (1 + math.cos(math.pi * (t - warmup_iters) / (cosine_cycle_iters - warmup_iters))) / 2
    else:
        return alpha_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float = 1.0):
    eps = 1e-6
    # compute the l2 norm of the gradients
    cat_params = []
    for p in parameters:
        if p.grad is not None:
            cat_params.append(p.grad.data.view(-1))
    cat_params = torch.cat(cat_params)
    l2_norm = torch.norm(cat_params, 2)
    
    if l2_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data = p.grad.data * (max_l2_norm / (l2_norm + eps))

    return l2_norm.item()

def data_loading(dataset: np.ndarray, batch_size: int, context_length: int, device: str = 'cpu'):
    possible_starting_indices = len(dataset) - context_length 
    starting_indices = np.random.randint(0, possible_starting_indices, size=batch_size)
    offsets = np.arange(context_length)
    final_indices = starting_indices.reshape(-1, 1) + offsets.reshape(1, -1)

    data = torch.tensor(dataset[final_indices], device=device).long()
    data_next = torch.tensor(dataset[final_indices+1], device=device).long()
    return (data, data_next)

def data_loading_val(dataset: np.ndarray, batch_size: int, context_length: int, device: str = 'cpu'):
    current_idx = 0
    starting_indices = np.arange(0, batch_size, 1)
    offsets = np.arange(context_length)
    while current_idx < len(dataset) - context_length - 1:
        final_indices = starting_indices.reshape(-1, 1) + offsets.reshape(1, -1)
        data = torch.tensor(dataset[final_indices], device=device).long()
        data_next = torch.tensor(dataset[final_indices+1], device=device).long()
        yield (data, data_next)
        current_idx += batch_size
        starting_indices += batch_size

def save_checkpoint(model, optimizer, iteration, out):
    # recover model state
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    # save model state
    torch.save({
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'iteration': iteration
    }, out)
    return 

def load_checkpoint(src, model, optimizer):
    # load model state
    model_state = torch.load(src)['model_state_dict']
    optimizer_state = torch.load(src)['optimizer_state_dict']
    iteration = torch.load(src)['iteration']
    # load model state
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return iteration


def transformer_decoder(model, tokenizer, prompt, device, temperature = 1.0, max_length = 100, top_p_threshold = 0.9):
    model.eval()
    eot_token = "<|endoftext|>"
    encoded_eot = tokenizer.encode(eot_token)[0]
    with torch.no_grad():
        token_ids = tokenizer.encode(prompt)
        token_ids = torch.tensor(token_ids, device=device)
        tokens_counter = 0
        while tokens_counter < max_length and token_ids[-1].item() != encoded_eot:
            output = model(token_ids.unsqueeze(0))
            output = output[0, -1, :]
            output = output / temperature
            # softmax
            max_output = torch.max(output, dim=-1, keepdim=True)[0]
            exp_output = torch.exp(output - max_output)
            sum_exp_output = torch.sum(exp_output, dim=-1, keepdim=True)
            output = exp_output / sum_exp_output

            # top-p sampling    
            sorted_probs, sorted_indices = torch.sort(output, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=0)
            bool_indices = cum_probs < top_p_threshold
            cutoff = bool_indices.sum()
            bool_indices[min(cutoff, len(bool_indices)-1)] = True

            sorted_probs = sorted_probs*bool_indices
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            # sample
            idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices[idx]
            token_ids = torch.cat([token_ids, next_token], dim=-1)
            tokens_counter += 1

        return tokenizer.decode(token_ids.tolist())



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.98), eps=1e-9, weight_decay=0.00):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if betas[0] < 0 or betas[0] > 1:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if betas[1] < 0 or betas[1] > 1:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "gamma": weight_decay}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] 
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            gamma = group["gamma"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] 
                t = state.get("t", 1) 
                grad = p.grad.data 

                # compute updated m and v
                m = beta1 * state.get("m", torch.zeros_like(p)) + (1 - beta1) * grad
                v = beta2 * state.get("v", torch.zeros_like(p)) + (1 - beta2) * grad**2

                # adjusted lr
                lrt = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                # update param
                p.data -= lrt*m / (torch.sqrt(v) + eps)
                # weight decay
                p.data = p.data*(1 - gamma*lr)
                # update state
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1 
        return loss

class Trainer():
    def __init__(self, model, **kwargs):
        self.model = model
        
        # Training data paths
        self.train_data = kwargs.get('train_data')
        self.val_data = kwargs.get('val_data')
        
        # Training hyperparameters
        self.num_iters = kwargs.get('num_iters', 5000)
        self.batch_size = kwargs.get('batch_size', 32)
        self.val_batch_size = kwargs.get('val_batch_size', 128)
        self.context_length = kwargs.get('context_length', 256)
        self.device = kwargs.get('device', 'cpu')
        self.eval_freq = kwargs.get('eval_freq', 1000)
        self.save_freq = kwargs.get('save_freq', 1000)
        self.checkpoint_dir = kwargs.get('checkpoint_dir', None)
        
        # Learning rate scheduler parameters
        self.alpha_max = kwargs.get('alpha_max', 1e-3)
        self.alpha_min = kwargs.get('alpha_min', 1e-4)
        self.warmup_iters = kwargs.get('warmup_iters', 1000)
        self.cosine_cycle_iters = kwargs.get('cosine_cycle_iters', 10000)
        self.max_l2_norm = kwargs.get('max_l2_norm', 1.0)

        self.scheduler = learning_rate_schedule
        self.gc = gradient_clipping
        self.loss_fn = cross_entropy_loss

        # AdamW optimizer parameters
        self.adamw_weight_decay = kwargs.get('adamw_weight_decay', 0.00)
        self.adamw_betas = kwargs.get('adamw_betas', (0.9, 0.98))
        self.adamw_eps = kwargs.get('adamw_eps', 1e-9)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-1, betas=self.adamw_betas, eps=self.adamw_eps, weight_decay=self.adamw_weight_decay)

        self.start_iter = 0
        
        if self.checkpoint_dir is not None:
            # get the latest checkpoint
            checkpoint_dir_path = pathlib.Path(self.checkpoint_dir)
            checkpoint_files = list(checkpoint_dir_path.glob('checkpoint_*.bin'))
            if checkpoint_files:
                # load the only checkpoint file
                only_checkpoint = checkpoint_files[0]
                self.start_iter = load_checkpoint(only_checkpoint, self.model, self.optimizer)

    def validate(self, val_dataset):
        self.model.eval()
        num_eval_batches = 0
        total_val_loss = 0.0
        print("number of validation batches: ", (len(val_dataset) - self.context_length - 1) // self.val_batch_size)
        print("batch size: ", self.val_batch_size)
        with torch.no_grad():
            for val_batch, val_batch_next in tqdm(data_loading_val(val_dataset, self.val_batch_size, self.context_length, self.device), desc="Validation"):
                output = self.model(val_batch)
                batch_loss = self.loss_fn(output, val_batch_next)
                total_val_loss += batch_loss.item()
                num_eval_batches += 1
                if num_eval_batches >= 30:
                    break
            val_loss = total_val_loss / num_eval_batches
            return val_loss
    
    
    def train(self):
        import time
        from collections import defaultdict
        
        print("=" * 60)
        print("TRAINING CONFIG")
        print("=" * 60)
        print(f"  num_iters: {self.num_iters}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  context_length: {self.context_length}")
        print(f"  device: {self.device}")
        print(f"  alpha_max: {self.alpha_max}")
        print(f"  alpha_min: {self.alpha_min}")
        print(f"  warmup_iters: {self.warmup_iters}")
        print(f"  cosine_cycle_iters: {self.cosine_cycle_iters}")
        print(f"  max_l2_norm: {self.max_l2_norm}")
        print(f"  eval_freq: {self.eval_freq}")
        print(f"  save_freq: {self.save_freq}")
        print(f"  adamw_weight_decay: {self.adamw_weight_decay}")
        print(f"  adamw_betas: {self.adamw_betas}")
        print(f"  adamw_eps: {self.adamw_eps}")
        print("=" * 60)
        
        print("Starting training...")
        train_dataset = np.load(self.train_data, mmap_mode='r')
        val_dataset = np.load(self.val_data, mmap_mode='r')
        print(f"Train dataset size: {len(train_dataset):,} tokens")
        print(f"Val dataset size: {len(val_dataset):,} tokens")
        print("Loaded train and validation datasets")
        
        # Profiling accumulators
        timings = defaultdict(float)
        profile_freq = 1000  # Print profiling every N iterations
        
        # add tqdm to the for loop
        for iter in tqdm(range(self.start_iter, self.num_iters), desc="Training"):
            iter_start = time.perf_counter()
            
            self.model.train()
            
            # Data loading
            t0 = time.perf_counter()
            train_data_batch, train_data_next_batch = data_loading(train_dataset, self.batch_size, self.context_length, self.device)
            timings['data_loading'] += time.perf_counter() - t0
            
            # Forward pass
            t0 = time.perf_counter()
            output = self.model(train_data_batch.to())
            timings['forward'] += time.perf_counter() - t0
            
            # Loss computation
            t0 = time.perf_counter()
            loss = self.loss_fn(output, train_data_next_batch)
            timings['loss'] += time.perf_counter() - t0
            
            # Backward pass
            t0 = time.perf_counter()
            self.optimizer.zero_grad()
            loss.backward()
            timings['backward'] += time.perf_counter() - t0
    
            # Gradient clipping
            t0 = time.perf_counter()
            l2_norm = self.gc(self.model.parameters(), self.max_l2_norm)
            timings['grad_clip'] += time.perf_counter() - t0
            
            # Optimizer step
            t0 = time.perf_counter()
            self.optimizer.step()
            timings['optimizer'] += time.perf_counter() - t0
            
            # Scheduler update
            t0 = time.perf_counter()
            new_lr = self.scheduler(iter, self.alpha_max, self.alpha_min, self.warmup_iters, self.cosine_cycle_iters)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            timings['scheduler'] += time.perf_counter() - t0
            
            timings['total'] += time.perf_counter() - iter_start
            
            # Print profiling summary
            if (iter + 1) % profile_freq == 0:
                n = profile_freq
                print(f"\n--- Profiling (last {n} iters) ---")
                print(f"  data_loading: {timings['data_loading']/n*1000:.2f} ms/iter")
                print(f"  forward:      {timings['forward']/n*1000:.2f} ms/iter")
                print(f"  loss:         {timings['loss']/n*1000:.2f} ms/iter")
                print(f"  backward:     {timings['backward']/n*1000:.2f} ms/iter")
                print(f"  grad_clip:    {timings['grad_clip']/n*1000:.2f} ms/iter")
                print(f"  optimizer:    {timings['optimizer']/n*1000:.2f} ms/iter")
                print(f"  scheduler:    {timings['scheduler']/n*1000:.2f} ms/iter")
                print(f"  TOTAL:        {timings['total']/n*1000:.2f} ms/iter")
                print(f"  throughput:   {n/timings['total']:.1f} iters/sec")
                timings.clear()

            # save checkpoint
            if iter % self.save_freq == 0:
                save_checkpoint(self.model, self.optimizer, iter, self.checkpoint_dir)
            if iter % self.eval_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    num_eval_batches = 10  
                    total_val_loss = 0.0
                    
                    for _ in range(num_eval_batches):
                        val_batch, val_batch_next = data_loading(val_dataset, self.batch_size, self.context_length, self.device)
                        output = self.model(val_batch)
                        batch_loss = self.loss_fn(output, val_batch_next)
                        total_val_loss += batch_loss.item()
                    
                    val_loss = total_val_loss / num_eval_batches
                    print(f"Validation loss: {val_loss:.4f}")

                wandb.log({
                'val_loss': val_loss,
                }, step=iter)

                
            # log training metrics wandb
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'l2_norm': l2_norm,
            }, step=iter)
            # log validation metrics wandb


if __name__ == "__main__":
    base_path = "./cs336_basics"
    with open(f"{base_path}/configs/lr_configs.json", "r") as f:
        lr_configs = json.load(f)
    
    # 'very_conservative','conservative', 'balanced', 'aggressive', 'very_aggressive'
    selected_strategy = "balanced" 
    lr_config = lr_configs[selected_strategy]
    
    # Configuration dictionary with all parameters
    config = {
        # Model parameters
        "d_model": 512,
        "num_heads": 16,
        "d_ff": 1344,
        "num_layers": 4,
        "vocab_size": 10000,
        "context_length": 256,
        
        # Training parameters
        "batch_size": 32,
        "num_iters": 5000,
        "eval_freq": 100,
        "save_freq": 1000,
        "device": "cpu",
        "val_batch_size": 128,
        
        # Data paths
        "train_data": f"{base_path}/outputs/TinyStories/train_dev_ids.npy",
        "val_data": f"{base_path}/outputs/TinyStories/valid_dev_ids.npy",
        
        # Tokenizer paths
        "vocab_filepath": f"{base_path}/outputs/TinyStories/vocab.pkl",
        "merges_filepath": f"{base_path}/outputs/TinyStories/merges.pkl",
        "special_tokens": ["<|endoftext|>"],
        
        "alpha_max": lr_config["alpha_max"],
        "alpha_min": lr_config["alpha_min"],
        "warmup_iters": int(lr_config["warmup_iters"]),
        "cosine_cycle_iters": int(lr_config["cosine_cycle_iters"]),
        "max_l2_norm": 1.0,

        # AdamW optimizer parameters    
        "adamw_weight_decay": 0.1,
        "adamw_betas": (0.9, 0.999),
        "adamw_eps": 1e-9,
    }

    checkpoint_dir = f"{base_path}/outputs/TinyStories/checkpoints/checkpoint_batch_{config['batch_size']}_iters_{config['num_iters']}_lr_{config['alpha_max']}_{config['alpha_min']}_{config['warmup_iters']}_cosine_{config['cosine_cycle_iters']}"
    config["checkpoint_dir"] = checkpoint_dir
    # Compute derived values
    config["total_tokens_processed"] = config["num_iters"] * config["batch_size"] * config["context_length"]

    # Debug: verify LR scheduler params
    print(f"LR Schedule: warmup_iters={config['warmup_iters']}, cosine_cycle_iters={config['cosine_cycle_iters']}, alpha_max={config['alpha_max']}")
    
    # Initialize wandb with config
    wandb.init(project="cs336_basics", name="transformer")
    wandb.config.update(config)

    # Load tokenizer
    tokenizer = Tokenizer.from_files(
        config["vocab_filepath"], 
        config["merges_filepath"], 
        config["special_tokens"]
    )

    # Load model
    model = TransformerLM(
        config["vocab_size"], 
        config["context_length"], 
        config["d_model"], 
        config["num_heads"], 
        config["d_ff"], 
        config["num_layers"]
    )
    model.to(config["device"])
    
    model = torch.compile(model, backend='eager')
    # print number of parameters in model
    print(f"Number of parameters in model: {sum(p.numel() for p in model.parameters())}")


    # set torch float32 precision for GPU optimization
    if config["device"] == 'cuda':
        torch.set_float32_matmul_precision('high')  # Enables TensorFloat32 for faster matmuls
    trainer = Trainer(model, **config)

    # load trainer
    trainer.train()

    # # evaluate model
    # load_checkpoint(checkpoint_dir, model, trainer.optimizer)
    # prompt = "Once upon a time"

    # output = transformer_decoder(model, tokenizer, prompt, config["device"])
    # print(output)

    # compute validation loss
    # load_checkpoint(checkpoint_dir, model, trainer.optimizer)
    # val_dataset = np.load(config["val_data"], mmap_mode='r')
    # val_loss = trainer.validate(val_dataset)
    # print(f"Validation loss: {val_loss:.4f}")

