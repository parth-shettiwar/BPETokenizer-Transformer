Problem (unicode1): Understanding Unicode (1 point)
(a) What Unicode character does chr(0) return?
'\x00'
(b) How does this character’s string representation (__repr__()) differ from its printed representation?
'\x00' vs Null character
(c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
string representation
'this is a test\x00string'
Print: 
this is a teststring

Problem (unicode2): Unicode Encodings
What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
Answer:
UTF-8 is more space efficient for texts that are primarily in ASCII, as it uses one byte for
ASCII characters, while UTF-16 and UTF-32 use two and four bytes respectively. UTF-8 is also
backward compatible with ASCII, making it easier to handle legacy systems. Additionally, UTF-8 is widely used on the web and in many programming languages, making it a more practical choice for interoperability.

Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
return "".join([bytes([b]).decode("utf-8") for b in bytestring])
Answer:
こ
The function is incorrect because it decodes each byte separately as its own UTF-8 sequence instead of decoding the full byte string at once, so any character encoded with multiple UTF-8 bytes (like é) will cause a UnicodeDecodeError or incorrect characters instead of the intended single Unicode character.

(c) Give a two byte sequence that does not decode to any Unicode character(s).
Answer:
b'\xe3\x81'
list(b'\xe3\x81') = [227, 129]
The byte 0xE3 indicates the start of a 3-byte UTF-8 character, which must be followed by two continuation bytes. However, in the sequence b'\xe3\x81', there is only one following byte, and although 0x81 is a valid continuation byte, the sequence is still incomplete because a third continuation byte is missing. Since UTF-8 does not allow truncated or partial multi-byte characters, this 2-byte sequence cannot be decoded into any valid Unicode character and will always raise a decode error.


Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)
(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size
of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary.
Serialize the resulting vocabulary and merges to disk for further inspection. How many hours
and memory did training take? What is the longest token in the vocabulary? Does it make sense?
Resource requirements: ≤30 minutes (no GPUs), ≤30GB RAM
Hint You should be able to get under 2 minutes for BPE training using multiprocessing during
pretokenization and the following two facts:
(a) The <|endoftext|> token delimits documents in the data files.
(b) The <|endoftext|> token is handled as a special case before the BPE merges are applied.
============================================================
TRAINING STATS
============================================================
Total training time: 98.65 seconds 
Peak memory usage: 10.8961 GB 
Longest token: ' accomplishment' (15 bytes)
============================================================

(b) Profile your code. What part of the tokenizer training process takes the most time?
Pretokenization took the most time, followed by merging loop.
============================================================
PROFILING SUMMARY
============================================================
Read file:                    4.17s  (  4.2%)
Split special tokens:         3.52s  (  3.6%)
Pretokenize:                 69.77s  ( 71.0%)
Vocab init:                   0.00s  (  0.0%)
Merge loop (total):          20.86s  ( 21.2%)
  - Avg per merge:          0.0021s
  - Min merge time:         0.0003s
  - Max merge time:         0.1280s
  - Total merges:             9743
------------------------------------------------------------
TOTAL TIME:                  98.32s
============================================================


Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
(a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary
size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What
is the longest token in the vocabulary? Does it make sense?
Resource requirements: ≤12 hours (no GPUs), ≤100GB RAM
============================================================
PROFILING SUMMARY
============================================================
Read file:                   46.18s  (  0.9%)
Split special tokens:        82.32s  (  1.6%)
Pretokenize:                699.33s  ( 13.5%)
Vocab init:                   0.00s  (  0.0%)
Merge loop (total):        4367.81s  ( 84.1%)
  - Avg per merge:          0.1374s
  - Min merge time:         0.0041s
  - Max merge time:        50.5794s
  - Total merges:            31743
------------------------------------------------------------
TOTAL TIME:                5195.65s
============================================================

============================================================
TRAINING STATS
============================================================
Total training time: 5221.81 seconds (1.4505 hours)
Peak memory usage: 26.2861 GB (26917.02 MB)
Longest token: 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ' (64 bytes) -> 4679 times in the dataset
============================================================

(b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
Deliverable: The merge loop takes more time now since vocab size is huge as well as dataset is much larger.

Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyS-
tories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these
sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
TinyStories compression ratio: 4.1123 bytes/token
OWT compression ratio: 4.6912 bytes/token

(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Com-
pare the compression ratio and/or qualitatively describe what happens.
total_bytes_owt_ts, total_tokens_owt_ts
The tiny stories tokenizer is much less efficient on owt docs than ts docs. This is because the tiny stories tokenizer is trained on a much smaller dataset and has a much smaller vocabulary size. OpenWebText contains news articles, technical content, and diverse topics with words rarely seen in children's stories. Words common in OWT but rare in TinyStories (e.g., "cryptocurrency", "legislation", "algorithm") won't have dedicated tokens. They get split into many smaller subword pieces, increasing token count. The 10K vocab can't capture the lexical diversity of web text. OWT's 32K tokenizer learned merges for technical terms, proper nouns, and domain-specific vocabulary.


(c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to
tokenize the Pile dataset (825GB of text)?
Time taken to encode OWT docs: 0.08 seconds
Throughput of OWT tokenizer: 390473.80 bytes/second
Time taken to encode TinyStories docs: 0.03 seconds
Throughput of TS tokenizer: 291510.71 bytes/second

Time taken to tokenize Pile dataset with TS tokenizer: 35.17 days
Time taken to tokenize Pile dataset with OWT tokenizer: 26.26 days

(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and devel-
opment datasets into a sequence of integer token IDs. We’ll use this later to train our language
model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is
uint16 an appropriate choice?
Range is sufficient: uint16 can represent values from 0 to 65,535. Our vocabulary sizes are:
TinyStories: 10,000 tokens
OpenWebText: 32,000 tokens
Both fit comfortably within the 65,535 max value of uint16.
Memory efficiency: uint16 uses 2 bytes per token

Problem (transformer_accounting): Transformer LM resource accounting (5 points): Refer sheet: https://docs.google.com/spreadsheets/d/1Rl0c0pFwpkKEoTXUP5EMbv3ZgZEMwPsn/edit?usp=sharing&ouid=109510744950242843494&rtpof=true&sd=true
(a) Consider GPT-2 XL, which has the following configuration:
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
27
num_heads : 25
d_ff : 6,400
Suppose we constructed our model using this configuration. How many trainable parameters
would our model have? Assuming each parameter is represented using single-precision floating
point, how much memory is required to just load this model?
Answer: For FP32 (single-precision), each parameter takes 4 bytes.
So memory to load the model weights is: 2,127,057,600×4=8,508,230,400 bytes
That is: ~8.51 GB. So we need ~8.5 GB (~7.9 GiB) just to store the parameters (not counting optimizer states, activations, KV cache, etc.).

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped
model. How many FLOPs do these matrix multiplies require in total? Assume that our input
sequence has context_length tokens.
Deliverable: A list of matrix multiplies (with descriptions), and the total number of FLOPs
required.
1. QKV Projections: N × 6 × L × D²
2. Attention score: QK^T: N × 2 × L² × D
3. Weighted Sum (AV): N × 2 × L² × D
4. Output Projection: N × 2 × L × D²
5. Gate & Value Proj (W1,W3): N × 4 × L × D × D_ff
6. Output Proj (W2): N × 2 × L × D × D_ff
7. Output Linear: N × L × D × V
Total: 4,513,336,524,800 FLOPs

(c) Based on your analysis above, which parts of the model require the most FLOPs?
Gate and value projection in FFN, require about 44% of the total FLOPs.

(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24
layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the
model size increases, which parts of the Transformer LM take up proportionally more or less of
the total FLOPs?
Deliverable: For each model, provide a breakdown of model components and its associated
FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a
one-to-two sentence description of how varying the model size changes the proportional FLOPs
of each component.
![alt text](./Images/gpt_flops.png)
As we increase model size, attention flops and AV sum component start decreasing and FFN flops start increasing. The QKV projections and other linear projections like output projection also increases.
(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one
forward pass change? How do the relative contribution of FLOPs of the model components
change?
FLOPs increase by 33x
![alt text](./Images/context_length_gpt.png)
As we increase context length, the QKV projections and other linear projections contribution decrease and attention scoring increases a lot. This is owing to L² growth for attention scoring.


Problem (learning_rate_tuning): Tuning the learning rate (1 point)
As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s
see that in practice in our toy example. Run the SGD example above with three other values for the
learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each
of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of
training)?
![alt text](./Images/1e1.png)
![alt text](./Images/1e2.png)
![alt text](./Images/1e3.png)
LR = 1e1 and 1e2 converge, while LR = 1e3 diverges. Between 1e1 and 1e2, 1e2 converges faster, while 1e1 didnt even converge in 10 iterations. 


Problem (adamwAccounting): Resource accounting for training with AdamW (2 points)
Let us compute how much memory and compute running AdamW requires. Assume we are using
float32 for every tensor.
(a) How much peak memory does running AdamW require? Decompose your answer based on the
memory usage of the parameters, activations, gradients, and optimizer state. Express your answer
in terms of the batch_size and the model hyperparameters (vocab_size, context_length,
num_layers, d_model, num_heads). Assume d_ff = 4 ×d_model.
For simplicity, when calculating memory usage of activations, consider only the following compo-
nents:
• Transformer block
– RMSNorm(s)
– Multi-head self-attention sublayer: QKV projections, Q⊤ K matrix multiply, softmax,
weighted sum of values, output projection.
– Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
• final RMSNorm
• output embedding
• cross-entropy on logits
Deliverable: An algebraic expression for each of parameters, activations, gradients, and opti-
mizer state, as well as the total.
(b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on
the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?
Deliverable: An expression that looks like a·batch_size + b for numerical values a, b, and a
number representing the maximum batch size.
(c) How many FLOPs does running one step of AdamW take?
Deliverable: An algebraic expression, with a brief justification.

For parts a,b,c, refer sheet: sheet: https://docs.google.com/spreadsheets/d/1Rl0c0pFwpkKEoTXUP5EMbv3ZgZEMwPsn/edit?pli=1&gid=1653816166#gid=1653816166
Max batch size ==> 30.49 + B*8.59 <= 80GB ==> B <= 5.76 ==> Max batch size = 5

(d) Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second)
relative to the hardware’s theoretical peak FLOP throughput [Chowdhery et al., 2022]. An
NVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assuming
you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a
batch size of 1024 on a single A100? Following Kaplan et al. [2020] and Hoffmann et al. [2022],
assume that the backward pass has twice the FLOPs of the forward pass.
GPT_absolute_flops_per_step = 13568190859904
expected_throughput = 19.5 * 10**12 * 0.5
Time_1step = GPT_absolute_flops_per_step / expected_throughput
Time_400Ksteps = Time_1step * 400000
Time_400Ksteps_days = Time_400Ksteps / (24 * 60 * 60)
Time_400Ksteps_days = 6.442635735946818 days
