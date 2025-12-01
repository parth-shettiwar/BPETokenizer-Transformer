import regex as re
# multiprocessing
import multiprocessing
import pathlib
from tqdm import tqdm
# Write a function that, given a path to an input text file, trains a (byte-level) BPE
# tokenizer. Your BPE training function should handle (at least) the following input parameters: 
# input_path, vocab_size, special_tokens. Return vocab: dict[int, bytes] and merges: list[tuple[bytes, bytes]]

# split by special tokens before pretokenization. This can be done using re.split with "|" ⌋.join(special_tokens). with careful use of re.escape since | may occur in the special tokens. example: re.split(r"[,.!?]", "Hello, world! How are you?")
do_print = False

class BPETrainer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pairs_bytes = {}
        self.pretokenized_tokens = {}
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

    def split_by_special_tokens(self, text: str, special_tokens: list[str]) -> list[str]:
        if not special_tokens:
            return [text]
        split_pattern = "|".join(re.escape(token) for token in special_tokens)
        return re.split(split_pattern, text)

    # Pretokenization function using the regex pattern provided in the assignment description. Return example:  {(108, 111, 119): 5 ...}
    def pretokenize_text(self, text: str, special_tokens: list[str]) -> dict[tuple[int], int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # use re.finditer
        pretokenized = [match.group(0) for match in re.finditer(PAT, text)]
        frequency = {}
        for token in pretokenized:
            # token_bytes = token.encode("utf-8")        
            # tuple_token = tuple(token_bytes)
            split_token_by_bytes = tuple([bytes([b]) for b in token.encode("utf-8")])
            frequency[split_token_by_bytes] = frequency.get(split_token_by_bytes, 0) + 1

        return frequency

    def pretokenize_multiprocessing(self, chunks: list[str], special_tokens: list[str]) -> dict[tuple[bytes], int]:
        with multiprocessing.Pool() as pool:
            results = pool.starmap(self.pretokenize_text, [(chunk, special_tokens) for chunk in chunks])
        # merge all dicts
        pretokenized_tokens = {}
        for result in results:
            for token, freq in result.items():
                pretokenized_tokens[token] = pretokenized_tokens.get(token, 0) + freq
        self.pretokenized_tokens =  pretokenized_tokens

    # go through each pair and then merge the most frequent pair only once. return merges and vocab
    def merge_pairs(self, vocab_size: int, vocab: dict[int, bytes], pairs_bytes: dict[tuple[bytes], int] | None = None) -> tuple[dict[int, bytes], tuple[bytes, bytes] | None]:
        if do_print:
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("pretokenized_tokens sample:", list(self.pretokenized_tokens.items()))
        
        most_frequent_pair = None

        if do_print:
            print("self.pairs_bytes before:", self.pairs_bytes)
        # check if pairs_bytes is {}
        if self.pairs_bytes == {}:
            for token_tuple, freq in self.pretokenized_tokens.items():
                # get all adjacent pairs
                for i in range(len(token_tuple) - 1):
                    # print("DEBUG token_tuple:", token_tuple)
                    pair = (token_tuple[i], token_tuple[i+1])
                    self.pairs_bytes[pair] = self.pairs_bytes.get(pair, 0) + freq
                    if most_frequent_pair is None or self.pairs_bytes[pair] > self.pairs_bytes.get(most_frequent_pair, 0) or (self.pairs_bytes[pair] == self.pairs_bytes.get(most_frequent_pair, 0) and pair > most_frequent_pair):
                        most_frequent_pair = pair
        else:
            # just find the most frequent pair
            most_frequent_pair = max(
                self.pairs_bytes.items(),
                key=lambda x: (x[1], x[0])
            )[0]


        if do_print:
            print("self.pairs_bytes", self.pairs_bytes)
        
        if most_frequent_pair is None or self.pairs_bytes.get(most_frequent_pair, 0) == 0:
            return vocab, None

        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        if do_print:
            print("most_frequent_pair", most_frequent_pair, new_token)
        # print top 10 frequent pairs

        if len(vocab) < vocab_size:
            vocab[len(vocab)] = new_token
        # update pretokenized_tokens
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
        if do_print:        
            print("updated self.pairs_bytes:", self.pairs_bytes)
            
            print("new_pretokenized_tokens", self.pretokenized_tokens)
        return vocab, most_frequent_pair


def train_bpe(
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
    trainer = BPETrainer(input_path, vocab_size, special_tokens)
    text = trainer.read_file_stream(input_path)
    print("File read complete. Length:", len(text))
    print("Splitting by special tokens...")
    chunks = trainer.split_by_special_tokens(text, special_tokens)
    print("Splitting complete. Number of chunks:", len(chunks))
    print("Pretokenizing with multiprocessing...")
    trainer.pretokenize_multiprocessing(chunks, special_tokens)
    print("Pretokenization complete. Number of unique pretokenized tokens:", len(trainer.pretokenized_tokens)) # pretokenized_tokens
    # Initialize vocab with single byte tokens
    vocab = {}
    for b in range(256):
        vocab[b] = bytes([b])
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        if st_bytes not in vocab.values() and len(vocab) < vocab_size:
            vocab[len(vocab)] = st_bytes
    print("Vocab initialized. Size:", len(vocab))

    # print("Starting BPE merges...")
    merges = []
    num_merges = vocab_size - len(vocab)
    count = 0
    with tqdm(total=num_merges, desc="BPE merges", unit="merge") as pbar:
        while len(vocab) < vocab_size:
            vocab, most_frequent_pair = trainer.merge_pairs(vocab_size, vocab)
            if most_frequent_pair is None or count ==1e12:
                print("No more pairs to merge.")
                break
            merges.append(most_frequent_pair)
            pbar.update(1)
            pbar.set_postfix({"vocab_size": len(vocab), "pair": f"{most_frequent_pair[0][:10]}+{most_frequent_pair[1][:10]}"})
            count += 1
    return vocab, merges

if __name__ == '__main__':

    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    print("Final Vocab size:", len(vocab))
    print("Final Merges size:", len(merges))