# Day 3 - KV Cache

## Notes

Prefill takes more time compared to Decode
    bcuz its the first time model sees the tokens
        when using KV cache

KV cache scales linear, so no memory problem for lots of time

Attention Computation
    token -> Wq Wk Wv -> Q "looking for"
                         K "offers to match"
                         V "content"
        
Colocating Prefill and Decode in the same GPU -> resource contention
large prompt (prefill) pauses or slows down token generation (decode)
to solve: token -> prefill nodes -> Decode nodes -> tokens
                   high compute     high memory (for KV cache)

RAG (heavy prefill): more prefill nodes, less decode nodes
Chatbots: less prefill nodes, more decode nodes

## KV Cache: Cached vs Uncached Decoding

Environment
    Device: CUDA
    CUDA device: Tesla T4
    Python: 3.12.13
    PyTorch: 2.11.0+cu128
    Total runtime RAM: 12.67 GB
    Available runtime RAM before model loading: 9.66 GB

Model
    Model: distilbert/distilgpt2
    Parameters: 81,912,576
    Estimated parameter memory: 312.47 MB
    Model loading time: 10.2925 seconds
    Process RAM after model loading: 1062.49 MB

Inference Configuration
    Prompt: Artificial intelligence infrastructure is important because
    Input tokens: 7
    Maximum new tokens: 25
    Decoding strategy: Greedy
    Compared methods: Cached and uncached
    Warm-up runs: 2
    Measured runs: 5

KV Cache Inspection
    Cached layers: 6
    Cached tensors: 12
    Key tensor shape: (1, 12, 7, 64)
    Value tensor shape: (1, 12, 7, 64)
    Estimated initial KV cache memory: 0.2461 MB

Benchmark Results
  method  mean_latency_seconds  median_latency_seconds  minimum_latency_seconds  maximum_latency_seconds  mean_tokens_per_second  mean_step_latency_seconds
uncached                0.4098                  0.4726                   0.2001                   0.6144                 71.6913                     0.0164
  cached                0.2775                  0.2847                   0.2097                   0.3424                 93.2891                     0.0110

Cache Benefit
    Latency speedup: 1.48x
    Latency reduction: 32.29%
    Throughput improvement: 30.13%
    Mean cached prefill latency: 0.0142 s

Validation
    Cached and uncached outputs match: True

Generated Text
    Artificial intelligence infrastructure is important because it is a way to make the world a better place.