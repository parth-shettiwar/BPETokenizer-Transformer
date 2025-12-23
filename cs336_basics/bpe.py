from collections.abc import Iterable, Iterator
from collections import defaultdict
import json
import regex as re
# multiprocessing
import multiprocessing
import pathlib
from tqdm import tqdm
import os
import time
import resource  
import sys
import pickle
import numpy as np

do_print = False

def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split text by special tokens, removing the special tokens.
    """
    if not special_tokens:
        return [text]
    split_pattern = "|".join(re.escape(token) for token in special_tokens)
    return re.split(split_pattern, text)

def split_preserving_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split text by special tokens, preserving the special tokens.
    """
    if not special_tokens:
        return [text], [False]
    # order special tokens by length, longest first
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    split_pattern = "|".join(re.escape(token) for token in special_tokens)
    # find locations of special tokens by regex
    special_token_locations = [(match.start(), match.end()) for match in re.finditer(split_pattern, text)]
    if special_token_locations==[]:
        return [text], [False]
    # split text at special token locations
    chunks = []
    special_token_flag = []
    for i in range(len(special_token_locations)):
        start,end = special_token_locations[i]
        if i==0:
            if start>0:
                chunks.append(text[:start])
                special_token_flag.append(False)
            chunks.append(text[start:end])
            special_token_flag.append(True)
        else:
            prev_start, prev_end = special_token_locations[i-1]
            if start>prev_end:
                chunks.append(text[prev_end:start])
                special_token_flag.append(False)
            chunks.append(text[start:end])
            special_token_flag.append(True)
    if end<len(text):
        chunks.append(text[end:])
        special_token_flag.append(False)

    return chunks, special_token_flag

def pretokenize_text_with_frequency(text: str) -> dict[tuple[int], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # use re.finditer
    pretokenized = [match.group(0) for match in re.finditer(PAT, text)]
    frequency = {}
    for token in pretokenized:
        split_token_by_bytes = tuple([bytes([b]) for b in token.encode("utf-8")])
        frequency[split_token_by_bytes] = frequency.get(split_token_by_bytes, 0) + 1
    return frequency

def pretokenize_frequency_multiprocessing(chunks: list[str], batch_size: int = 40000, num_workers: int = 32) -> dict[tuple[bytes], int]:
    
    pretokenized_tokens = {}
    print(f"Processing {len(chunks)} chunks with {num_workers} workers")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i+batch_size]
            # Use imap_unordered with chunksize for better throughput
            for result in pool.imap_unordered(pretokenize_text_with_frequency, batch, chunksize=100):
                for token, freq in result.items():
                    pretokenized_tokens[token] = pretokenized_tokens.get(token, 0) + freq
    return pretokenized_tokens

# normal pretokenize function without computing frequency
def pretokenize_text(text: str) -> dict[tuple[bytes], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokenized = [match.group(0) for match in re.finditer(PAT, text)]
    final_split_token_by_bytes = []
    for token in pretokenized:
        split_token_by_bytes = tuple([bytes([b]) for b in token.encode("utf-8")])
        final_split_token_by_bytes.append(split_token_by_bytes)
    return final_split_token_by_bytes



class BPETrainer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pairs_bytes = defaultdict(int)
        self.pretokenized_tokens = {}
        self.vocab = {}
        self.most_frequent_pair = None
        self.pair_to_tokens = defaultdict(set)

    def read_file(self, input_path: str) -> str:
        with open(input_path, encoding="utf-8") as f:
            text = f.read()
        return text

    def read_file_stream(self, path):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                out.append(line)
        return "".join(out)

    def merge_pairs(self, vocab_size: int, pairs_bytes: dict[tuple[bytes], int] | None = None) -> tuple[bytes, bytes] | None:        
        most_frequent_pair = None

        if len(self.pairs_bytes) == 0:
            for token_tuple, freq in self.pretokenized_tokens.items():
                for i in range(len(token_tuple) - 1):
                    pair = (token_tuple[i], token_tuple[i+1])
                    self.pairs_bytes[pair] += freq
                    self.pair_to_tokens[pair].add(token_tuple)
                    if most_frequent_pair is None or self.pairs_bytes[pair] > self.pairs_bytes[most_frequent_pair] or (self.pairs_bytes[pair] == self.pairs_bytes[most_frequent_pair] and pair > most_frequent_pair):
                        most_frequent_pair = pair
        else:
            most_frequent_pair = max(
                self.pairs_bytes.items(),
                key=lambda x: (x[1], x[0])
            )[0]
        
        if most_frequent_pair is None or self.pairs_bytes[most_frequent_pair] == 0:
            return None

        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        if len(self.vocab) < vocab_size:
            self.vocab[len(self.vocab)] = new_token
        
        tokens_iterate = list(self.pair_to_tokens[most_frequent_pair])
        for token_tuple in tokens_iterate:
            freq = self.pretokenized_tokens[token_tuple]
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i+1])
                self.pairs_bytes[pair] -= freq
                self.pair_to_tokens[pair].discard(token_tuple)
                if self.pairs_bytes[pair] <= 0:
                    del self.pairs_bytes[pair]

            new_token_tuple = []
            i = 0

            while i < len(token_tuple):
                if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i+1]) == most_frequent_pair:
                    new_token_tuple.append(new_token)
                    i += 2
                else:
                    new_token_tuple.append(token_tuple[i])
                    i += 1
            new_token_tuple = tuple(new_token_tuple)
            del self.pretokenized_tokens[token_tuple]
            self.pretokenized_tokens[new_token_tuple] = (self.pretokenized_tokens.get(new_token_tuple, 0) + freq)

            for i in range(len(new_token_tuple) - 1):
                pair = (new_token_tuple[i], new_token_tuple[i+1])
                self.pairs_bytes[pair] += freq
                self.pair_to_tokens[pair].add(new_token_tuple)

        return most_frequent_pair


    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Train a byte-level BPE tokenizer on the input text file.

        Args:
            input_path (str): Path to the input text file.
            vocab_size (int): Desired vocabulary size.
            special_tokens (list[str]): List of special tokens to include in the vocabulary.

        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: A tuple containing the vocabulary
            as a dictionary mapping token IDs to byte sequences, and a list of merges as tuples
            of byte sequences.
        """
        # Profiling: track time for each major step
        profile_times = {}
        total_start = time.time()
        
        print("Reading file...")
        t0 = time.time()
        text = self.read_file_stream(self.input_path)
        profile_times['read_file'] = time.time() - t0
        print(f"File read complete. Length: {len(text)} (took {profile_times['read_file']:.2f}s)")
        
        print("Splitting by special tokens...")
        t0 = time.time()
        chunks = split_by_special_tokens(text, self.special_tokens)
        profile_times['split_special_tokens'] = time.time() - t0
        print(f"Splitting complete. Number of chunks: {len(chunks)} (took {profile_times['split_special_tokens']:.2f}s)")
        
        print("Pretokenizing with multiprocessing...")
        t0 = time.time()
        self.pretokenized_tokens = pretokenize_frequency_multiprocessing(chunks)
        profile_times['pretokenize'] = time.time() - t0
        print(f"Pretokenization complete. Number of unique pretokenized tokens: {len(self.pretokenized_tokens)} (took {profile_times['pretokenize']:.2f}s)")
        
        # Initialize vocab with single byte tokens
        print("Initializing vocabulary...")
        t0 = time.time()
        vocab = {}
        for b in range(256):
            vocab[b] = bytes([b])
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in vocab.values() and len(vocab) < self.vocab_size:
                vocab[len(vocab)] = st_bytes
        self.vocab = vocab
        profile_times['vocab_init'] = time.time() - t0
        print(f"Vocab initialized. Size: {len(vocab)} (took {profile_times['vocab_init']:.2f}s)")

        # BPE merge loop
        merges = []
        num_merges = self.vocab_size - len(self.vocab)
        count = 0
        merge_times = []
        
        print("Starting BPE merges...")
        merge_start = time.time()
        with tqdm(total=num_merges, desc="BPE merges", unit="merge") as pbar:
            while len(self.vocab) < self.vocab_size:
                t0 = time.time()
                most_frequent_pair = self.merge_pairs(self.vocab_size, self.vocab)
                merge_times.append(time.time() - t0)
                
                if most_frequent_pair is None:
                    print("No more pairs to merge.")
                    break
                merges.append(most_frequent_pair)
                pbar.update(1)
                pbar.set_postfix({"vocab_size": len(self.vocab), "pair": f"{most_frequent_pair[0][:10]}+{most_frequent_pair[1][:10]}"})
                count += 1
        
        profile_times['merge_loop_total'] = time.time() - merge_start
        profile_times['merge_avg'] = sum(merge_times) / len(merge_times) if merge_times else 0
        profile_times['merge_min'] = min(merge_times) if merge_times else 0
        profile_times['merge_max'] = max(merge_times) if merge_times else 0
        profile_times['total'] = time.time() - total_start
        
        # Print profiling summary
        print("\n" + "="*60)
        print("PROFILING SUMMARY")
        print("="*60)
        print(f"Read file:              {profile_times['read_file']:>10.2f}s  ({profile_times['read_file']/profile_times['total']*100:>5.1f}%)")
        print(f"Split special tokens:   {profile_times['split_special_tokens']:>10.2f}s  ({profile_times['split_special_tokens']/profile_times['total']*100:>5.1f}%)")
        print(f"Pretokenize:            {profile_times['pretokenize']:>10.2f}s  ({profile_times['pretokenize']/profile_times['total']*100:>5.1f}%)")
        print(f"Vocab init:             {profile_times['vocab_init']:>10.2f}s  ({profile_times['vocab_init']/profile_times['total']*100:>5.1f}%)")
        print(f"Merge loop (total):     {profile_times['merge_loop_total']:>10.2f}s  ({profile_times['merge_loop_total']/profile_times['total']*100:>5.1f}%)")
        print(f"  - Avg per merge:      {profile_times['merge_avg']:>10.4f}s")
        print(f"  - Min merge time:     {profile_times['merge_min']:>10.4f}s")
        print(f"  - Max merge time:     {profile_times['merge_max']:>10.4f}s")
        print(f"  - Total merges:       {len(merge_times):>10d}")
        print("-"*60)
        print(f"TOTAL TIME:             {profile_times['total']:>10.2f}s")
        print("="*60 + "\n")
        
        return self.vocab, merges



class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens 
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class
        method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        vocab: dict[int, bytes] - A dictionary mapping token IDs to byte sequences.
        merges: list[tuple[bytes, bytes]] - A list of merges as tuples of byte sequences.
        vocab and merge are in pickle format
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """

        reverse_vocab = {v: k for k, v in self.vocab.items()} 
               
        # split into chunks by special tokens
        chunks, special_token_flag = split_preserving_special_tokens(text, self.special_tokens)
        merges_rank_dic = {merge: rank for rank, merge in enumerate(self.merges)}
        token_ids = []
        print("Number of chunks: ", len(chunks))
      
        token_tuple_token_ids = {}
        for idx, chunk in enumerate(tqdm(chunks)):
               
            if special_token_flag[idx]:
                token_ids.append(reverse_vocab[chunk.encode("utf-8")])
                continue

            # pretokenize chunk
            pretokenized_tokens = pretokenize_text(chunk)
            # apply merges in same order as in self.merges.
            for token_tuple in pretokenized_tokens:
                if token_tuple in token_tuple_token_ids:
                    token_ids.extend(token_tuple_token_ids[token_tuple])
                    continue
                
                original_token_tuple = token_tuple
                token_tuple = list(token_tuple)
                while len(token_tuple)>1:
                    best_pos = -1
                    rank_min = float('inf')
                    for i in range(len(token_tuple) - 1):
                        if (token_tuple[i], token_tuple[i+1]) in merges_rank_dic:
                            if merges_rank_dic[(token_tuple[i], token_tuple[i+1])] < rank_min:
                                best_merge = bytes(token_tuple[i] + token_tuple[i+1])
                                rank_min = merges_rank_dic[(token_tuple[i], token_tuple[i+1])]
                                best_pos = i
                    if best_pos == -1:
                        break
        
                    token_tuple[best_pos] = best_merge
                    del token_tuple[best_pos+1]
                    

                current_token_ids = []
                for token in token_tuple:
                    if token in reverse_vocab:
                        current_token_ids.append(reverse_vocab[token])
                    else:
                        raise ValueError(f"Token {token} not in vocabulary.")
                token_tuple_token_ids[original_token_tuple] = current_token_ids
                token_ids.extend(current_token_ids)

        return token_ids

         

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that we cannot directly load into
        memory.
        """
        # cache token tuple to token id
        token_tuple_token_ids = {}
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        # get next string from iterable
        for text in iterable:
            chunks, special_token_flag = split_preserving_special_tokens(text, self.special_tokens)
            for i, chunk in enumerate(chunks):
                if special_token_flag[i]:
                    yield reverse_vocab[chunk.encode("utf-8")]
                    continue
                pretokenized_tokens = pretokenize_text(chunk)
                for token_tuple in pretokenized_tokens:
                    if token_tuple in token_tuple_token_ids:
                        for token_id in token_tuple_token_ids[token_tuple]:
                            yield token_id
                        continue
                    original_token_tuple = token_tuple

                    for merge in self.merges:
                        new_token_list = []
                        i = 0
                        while i < len(token_tuple):
                            if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i+1]) == merge:
                                new_token_list.append(token_tuple[i] + token_tuple[i+1])
                                i += 2
                            else:
                                new_token_list.append(token_tuple[i])
                                i += 1
                        token_tuple = tuple(new_token_list)
                    current_token_ids = []
                    for token in token_tuple:
                        if token in reverse_vocab:
                            current_token_ids.append(reverse_vocab[token])
                        else:
                            raise ValueError(f"Token {token} not in vocabulary.")
                    token_tuple_token_ids[original_token_tuple] = current_token_ids
                    for token_id in current_token_ids:
                        yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        try:
            token_bytes = b"".join([self.vocab[token_id] for token_id in ids])
            return token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_bytes = b"".join([self.vocab[token_id] for token_id in ids])
            return token_bytes.decode("utf-8", errors="replace")  # Replace invalid bytes with U+FFFD


def save_tokenizer(vocab, merges, output_folder):
    # Save vocab as pickle (bytes preserved exactly)
    with open(os.path.join(output_folder, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    
    # Save merges as pickle (list of byte tuples)
    with open(os.path.join(output_folder, "merges.pkl"), "wb") as f:
        pickle.dump(merges, f)

def train_dataset(input_path: str, vocab_size: int, special_tokens: list[str], output_save_folder: str):
    print("="*60)
    
    start_time = time.time()
    trainer = BPETrainer(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
    vocab, merges = trainer.train_bpe()
    
    # Get training stats
    end_time = time.time()
    # Get peak memory from OS (no overhead during training)
    peak_mem_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS ru_maxrss is in bytes, on Linux it's in KB
    
    if sys.platform == "darwin":
        peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    else:
        peak_mem_gb = peak_mem_bytes / (1024 ** 2)  # Linux: KB to GB
    
    total_time_seconds = end_time - start_time
    total_time_hours = total_time_seconds / 3600
    
    # Find longest token in vocabulary
    longest_token = max(vocab.values(), key=len)
    longest_token_str = longest_token.decode("utf-8", errors="replace")
    longest_token_len = len(longest_token)
    
    print("\n" + "="*60)
    print("TRAINING STATS")
    print("="*60)
    print(f"Total training time: {total_time_seconds:.2f} seconds ({total_time_hours:.4f} hours)")
    print(f"Peak memory usage: {peak_mem_gb:.4f} GB ({peak_mem_gb * 1024:.2f} MB)")
    print(f"Longest token: '{longest_token_str}' ({longest_token_len} bytes)")
    print("="*60 + "\n")

    # Serialize the resulting vocabulary and merges to disk
    # Convert bytes to strings for JSON (latin-1 is lossless for bytes 0-255)
    save_tokenizer(vocab, merges, output_save_folder)
    print(f"Saved vocabulary and merges to {output_save_folder}")


# if __name__ == "__main__":
    # Train Tiny stories dataset
    # input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    # vocab_size = 10000
    # special_tokens = ["<|endoftext|>"]
    # output_save_folder = "./cs336_basics/outputs/TinyStories"
    # train_dataset(input_path, vocab_size, special_tokens, output_save_folder)

    # Train OWT
    # input_path = "./data/owt_train.txt"
    # vocab_size = 32000
    # special_tokens = ["<|endoftext|>"]
    # output_save_folder = "./cs336_basics/outputs/owt"
    # train_dataset(input_path, vocab_size, special_tokens, output_save_folder)