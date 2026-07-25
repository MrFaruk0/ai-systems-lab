# Day 2 - Autoregressive Generation

## Notes

How a GPT works?
    take the untrained GPT 3
    generate training examples on sentences with sliding window -> unsupersived learning
    for each example;
        predict -> compute error rate -> update weights
    context window
    fine-tuning -> extra training the model for a specific task

Transformer Architecture
    Self-Attention
        for each word, find a weight for how much it must "pay attention" to each other word
        word embeddings -> Q, K, V -> dot product of Q and K -> softmax -> concat = 1 head
    Feedforward Neural Network

Embedding: models vocab, make number meaningful
Logit: a score for each token in the vocab 
       pass logits through softmax to get probabilites

## Manual Autoregressive Generation

Environment
    Device: CPU
    CUDA device: None
    Python: 3.12.13
    PyTorch: 2.11.0+cpu
    Total runtime RAM: 12.67 GB
    Available runtime RAM before model loading: 9.78 GB

Model
    Model: distilbert/distilgpt2
    Parameters: 81,912,576
    Estimated parameter memory: 312.47 MB
    Model loading time: 12.1020 seconds
    Process RAM after model loading: 594.23 MB

Generation Configuration
    Prompt: Artificial intelligence infrastructure is important because
    Input tokens: 7
    Maximum new tokens: 25
    Greedy decoding: Enabled
    Sampling temperatures: 0.2, 0.7, 1.0, 1.5, 2.0
    Warm-up runs: 2
    Measured runs: 5

Greedy Benchmark Results
    Mean latency: 3.1085 s
    Median latency: 2.9067 s
    Minimum latency: 2.8292 s
    Maximum latency: 3.5488 s
    Latency standard deviation: 0.2884 s
    Mean output throughput: 8.0424 tokens/s
    Mean per-token latency: 0.1243 s

Temperature Comparison
 temperature  mean_latency_seconds  tokens_per_second  top_probability  entropy
         0.2                3.1819             7.8568           1.0000   0.0005
         0.7                3.1166             8.0214           0.8447   0.8432
         1.0                3.1559             7.9218           0.5248   2.9244
         1.5                3.1714             7.8829           0.1114   7.3261
         2.0                3.0964             8.0740           0.0236   9.1384

Top 10 Next-Token Predictions
 rank         token  probability
    1         ' it'     0.524807
    2        ' the'     0.060650
    3         ' of'     0.055730
    4         ' we'     0.051196
    5      ' there'     0.018408
    6        ' its'     0.014488
    7           ','     0.012909
    8       ' they'     0.012694
    9 ' artificial'     0.009872
   10       ' this'     0.007541