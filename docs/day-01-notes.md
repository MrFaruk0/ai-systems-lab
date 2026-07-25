# Day 1 - Inference Basics

## Notes
Token ≈ 4 character

Prompt
 -> Tokenizer
    turns prompt into a sequence of int (vector)
 -> Prefill (model understands the prompt)
    tokenized input * Wq = Q
    tokenized input * Wk = K
    tokenized input * Wv = V (KV cache to speed up next token generation)
    attention heads compute attention
        multi-headed attention -> better text understanding
                                  a few head computes attention then concatenate and multiply with weight
 -> De-code (token generation)
    one token generated at a time using output from Prefill
    autoregressive, slowest phase, memory bound
    each output token is fed as input to generate next token
 -> De-tokenize
 Output

 Long Input Short Output -> Big Prefill, Smaller Decode - bottleneck is Prefill
 Long Input Long Output -> Big Prefill, Big Decode - bottleneck is both
 Short Input Long Output -> Smaller Prefill, Big Decode - bottleneck is Decode
 Short Input Short Output -> Small Prefill, Small Decode

## DistilGPT-2 Inference Benchmark
Environment
    Device: CPU
    CUDA device: None
    Python: 3.12.13
    PyTorch: 2.11.0+cpu
    Total runtime RAM: 12.67 GB
    Available runtime RAM before model loading: 8.21 GB
Model
    Model: distilbert/distilgpt2
    Parameters: 81,912,576
    Estimated parameter memory: 312.47 MB
    Model loading time: 0.9580 seconds
    Highest sampled process RAM during inference: 961.27 MB
Inference Configuration
    Prompt: Artificial intelligence infrastructure is important because
    Input tokens: 7
    Maximum new tokens: 25
    Decoding strategy: Greedy decoding (do_sample=False)
    Warm-up runs: 1
    Measured runs: 3
Results
    Metric	                    Value
    Mean latency	            1.1170 s
    Median latency	            1.1298 s
    Minimum latency	            1.0016 s
    Maximum latency	            1.2195 s
    Mean output throughput	    22.5296 tokens/s
    Highest sampled process RAM	961.27 MB
Generated Text
    Artificial intelligence infrastructure is important because it is a way to make the world a better place.
