from collections.abc import Iterable, Iterator
import json
import regex as re
# multiprocessing
import multiprocessing
import pathlib
from tqdm import tqdm

do_print = False

def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    split_pattern = "|".join(re.escape(token) for token in special_tokens)
    return re.split(split_pattern, text)

def pretokenize_text_with_frequency(text: str) -> dict[tuple[int], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # use re.finditer
    pretokenized = [match.group(0) for match in re.finditer(PAT, text)]
    frequency = {}
    for token in pretokenized:
        split_token_by_bytes = tuple([bytes([b]) for b in token.encode("utf-8")])
        frequency[split_token_by_bytes] = frequency.get(split_token_by_bytes, 0) + 1
    return frequency

def pretokenize_frequency_multiprocessing(chunks: list[str]) -> dict[tuple[bytes], int]:
    with multiprocessing.Pool() as pool:
        results = pool.starmap(pretokenize_text_with_frequency, [(chunk,) for chunk in chunks])
    # merge all dicts
    pretokenized_tokens = {}
    for result in results:
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
        self.pairs_bytes = {}
        self.pretokenized_tokens = {}
        self.vocab = {}
        self.most_frequent_pair = None
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

        if self.pairs_bytes == {}:
            for token_tuple, freq in self.pretokenized_tokens.items():
                for i in range(len(token_tuple) - 1):
                    pair = (token_tuple[i], token_tuple[i+1])
                    self.pairs_bytes[pair] = self.pairs_bytes.get(pair, 0) + freq
                    if most_frequent_pair is None or self.pairs_bytes[pair] > self.pairs_bytes.get(most_frequent_pair, 0) or (self.pairs_bytes[pair] == self.pairs_bytes.get(most_frequent_pair, 0) and pair > most_frequent_pair):
                        most_frequent_pair = pair
        else:
            most_frequent_pair = max(
                self.pairs_bytes.items(),
                key=lambda x: (x[1], x[0])
            )[0]
        
        if most_frequent_pair is None or self.pairs_bytes.get(most_frequent_pair, 0) == 0:
            return None

        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        if len(self.vocab) < vocab_size:
            self.vocab[len(self.vocab)] = new_token
        new_pretokenized_tokens = {}
        
        for token_tuple, freq in self.pretokenized_tokens.items():
            has_pair = False
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i+1])
                if pair == most_frequent_pair:
                    has_pair = True
                    break

            if has_pair:
                for i in range(len(token_tuple) - 1):
                    pair = (token_tuple[i], token_tuple[i+1])
                    self.pairs_bytes[pair] = self.pairs_bytes.get(pair, 0) - freq
                    if self.pairs_bytes[pair] <= 0:
                        del self.pairs_bytes[pair]

            new_token_tuple = []
            i = 0
            if has_pair:
                while i < len(token_tuple):
                    if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i+1]) == most_frequent_pair:
                        new_token_tuple.append(new_token)
                        i += 2
                    else:
                        new_token_tuple.append(token_tuple[i])
                        i += 1
                new_token_tuple = tuple(new_token_tuple)
                new_pretokenized_tokens[new_token_tuple] = (new_pretokenized_tokens.get(new_token_tuple, 0) + freq)
            else:
                new_pretokenized_tokens[token_tuple] = (new_pretokenized_tokens.get(token_tuple, 0) + freq)

            if has_pair:
                for i in range(len(new_token_tuple) - 1):
                    pair = (new_token_tuple[i], new_token_tuple[i+1])
                    self.pairs_bytes[pair] = self.pairs_bytes.get(pair, 0) + freq

        self.pretokenized_tokens = new_pretokenized_tokens

        return most_frequent_pair


    def train_bpe(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
        print("Reading file...")
        
        text = self.read_file_stream(input_path)
        print("File read complete. Length:", len(text))
        print("Splitting by special tokens...")
        chunks = split_by_special_tokens(text, special_tokens)
        print("Splitting complete. Number of chunks:", len(chunks))
        print("Pretokenizing with multiprocessing...")
        self.pretokenized_tokens = pretokenize_frequency_multiprocessing(chunks)
        print("Pretokenization complete. Number of unique pretokenized tokens:", len(self.pretokenized_tokens)) # pretokenized_tokens
        # Initialize vocab with single byte tokens
        vocab = {}
        for b in range(256):
            vocab[b] = bytes([b])
        for st in special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in vocab.values() and len(vocab) < vocab_size:
                vocab[len(vocab)] = st_bytes
        self.vocab = vocab
        print("Vocab initialized. Size:", len(vocab))

        # print("Starting BPE merges...")
        merges = []
        num_merges = vocab_size - len(self.vocab)
        count = 0
        with tqdm(total=num_merges, desc="BPE merges", unit="merge") as pbar:
            while len(self.vocab) < vocab_size:
                most_frequent_pair = self.merge_pairs(vocab_size, self.vocab)
                if most_frequent_pair is None:
                    print("No more pairs to merge.")
                    break
                merges.append(most_frequent_pair)
                pbar.update(1)
                pbar.set_postfix({"vocab_size": len(self.vocab), "pair": f"{most_frequent_pair[0][:10]}+{most_frequent_pair[1][:10]}"})
                count += 1
        return self.vocab, merges



class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens 

    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        """
        import json
        with open(vocab_filepath, "r") as f:
            vocab_json = json.load(f)
        
        vocab = {}
        for token_str, token_id in vocab_json.items():
            vocab[token_id] = token_str.encode("utf-8")

        # Read merges from text file
        # Format: each line is "token1 token2"
        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        first = parts[0].encode("utf-8")
                        second = parts[1].encode("utf-8")
                        merges.append((first, second))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        # cache token tuple to token id
        token_tuple_token_ids = {}
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        # pretokenize each chunk
        pretokenized_tokens = pretokenize_text(text)
        # apply merges in same order as in self.merges.
        token_ids = []
        print("pretokenized_tokens:", pretokenized_tokens)
        for token_tuple in pretokenized_tokens:
            print("########## Token tuple before merges:", token_tuple)
            if token_tuple in token_tuple_token_ids:
                token_ids.extend(token_tuple_token_ids[token_tuple])
                continue
            
            original_token_tuple = token_tuple
            for merge in self.merges:
                new_token_list = []
                i = 0
                while i < len(token_tuple):
                    if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i+1]) == merge:
                        new_token_list.append(token_tuple[i] + token_tuple[i+1])
                        i += 2
                        print("Applied merge:", merge)
                    else:
                        new_token_list.append(token_tuple[i])
                        i += 1
                token_tuple = tuple(new_token_list)
            current_token_ids = []
            for token in token_tuple:
                print("Token:", token)
                if token in reverse_vocab:
                    current_token_ids.append(reverse_vocab[token])
                else:
                    raise ValueError(f"Token {token} not in vocabulary.")
            token_tuple_token_ids[original_token_tuple] = current_token_ids
            token_ids.extend(current_token_ids)
            print("########## Token tuple after merges:", token_tuple, token_ids)
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
            pretokenized_tokens = pretokenize_text(text)
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
            return "".join([self.vocab[token_id].decode("utf-8") for token_id in ids])
        except KeyError:
            raise ValueError(f"Token IDs not in vocabulary.")

