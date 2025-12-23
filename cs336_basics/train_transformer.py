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

def data_loading(dataset: np.ndarray, batch_size: int, context_length: int, device: str = 'cpu'):
    possible_starting_indices = len(dataset) - context_length 
    starting_indices = np.random.randint(0, possible_starting_indices, size=batch_size)
    offsets = np.arange(context_length)
    final_indices = starting_indices.reshape(-1, 1) + offsets.reshape(1, -1)

    data = torch.tensor(dataset[final_indices], device=device).long()
    data_next = torch.tensor(dataset[final_indices+1], device=device).long()
    return (data, data_next)

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
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.999), eps=1e-8, weight_decay=0.1):
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
    def __init__(self, model, scheduler, train_data, val_data, num_iters, batch_size, context_length, device, checkpoint_dir, eval_freq = 1000, save_freq = 1000):
        self.model = model
        self.scheduler = scheduler
        self.train_data = train_data
        self.val_data = val_data
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = AdamW(self.model.parameters(), lr=1e-1)
        self.scheduler = learning_rate_schedule
        self.gc = gradient_clipping
        self.loss_fn = cross_entropy_loss

        # set lr scheduler parameters
        self.alpha_max = 1e-3
        self.alpha_min = 1e-4
        self.warmup_iters = 1000
        self.cosine_cycle_iters = 10000
        self.max_l2_norm = 1.0
        self.start_iter = 0
        
        if checkpoint_dir is not None:
            # get the latest checkpoint
            checkpoint_dir_path = pathlib.Path(checkpoint_dir)
            checkpoint_files = list(checkpoint_dir_path.glob('checkpoint_*.bin'))
            if checkpoint_files:
                # load the om;y checkpoint file
                only_checkpoint = checkpoint_files[0]
                self.start_iter = load_checkpoint(only_checkpoint, self.model, self.optimizer)


    def train(self):
        import time
        from collections import defaultdict
        
        print("Starting training...")
        train_dataset = np.load(self.train_data, mmap_mode='r')
        val_dataset = np.load(self.val_data, mmap_mode='r')
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
            self.gc(self.model.parameters(), self.max_l2_norm)
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
                # load data
                    val_data_batch, val_data_next_batch = data_loading(val_dataset, self.batch_size, self.context_length, self.device)
                    # forward pass
                    output = self.model(val_data_batch.to())
                    # backward pass
                    loss = self.loss_fn(output, val_data_next_batch)
                    print(f"Validation loss: {loss.item()}")

                wandb.log({
                'val_loss': loss.item(),
                })

                
            # log training metrics wandb
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
            })
            # log validation metrics wandb


if __name__ == "__main__":
    # define model parameters
    d_model = 512
    num_heads = 16
    d_ff = 1344
    num_layers = 4
    vocab_size = 10000
    context_length = 256

    batch_size = 32
    num_iters = 5000
    eval_freq = 10
    save_freq = 1000
    total_tokens_processed = num_iters*batch_size*context_length
    checkpoint_dir = "./cs336_basics/outputs/TinyStories/checkpoints"

    wandb.init(project="cs336_basics", name="transformer")
    wandb.config.update({
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "context_length": context_length,
    })

    device = 'cpu'

    # load train and validation data for TS
    train_data = "./cs336_basics/outputs/TinyStories/train_dev_ids.npy"
    val_data = "./cs336_basics/outputs/TinyStories/valid_dev_ids.npy"
    # train_data_ids = np.load(train_data)
    # val_data_ids = np.load(val_data)

    # load tokenizer
    vocab_filepath = "./cs336_basics/outputs/TinyStories/vocab.pkl"
    merges_filepath = "./cs336_basics/outputs/TinyStories/merges.pkl"
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    # load model
    model = TransformerLM(vocab_size, context_length, d_model, num_heads, d_ff, num_layers)
    model.to(device)
    
    # # Compile model for faster training (PyTorch 2.0+)
    # model = torch.compile(model)

    # load trainer
    trainer = Trainer(model, tokenizer, train_data, val_data, num_iters, batch_size, context_length, device, checkpoint_dir, eval_freq, save_freq)
    trainer.train()

