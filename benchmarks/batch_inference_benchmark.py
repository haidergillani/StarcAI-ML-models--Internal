import time
import numpy as np
from huggingface_hub import hf_hub_download
import onnxruntime
import sys
from pathlib import Path

# Add parent directory to path to import from onnx_inference
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from onnx_inference.tokenizer import BertTokenizer

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def initialize_model():
    """Initialize model and tokenizer."""
    # Download model and vocab
    model_path = hf_hub_download(
        repo_id="MSaadAsad/FinBERT-merged-tone-fls",
        filename="finbert_6layers_quantized_compat.onnx"
    )
    vocab_path = hf_hub_download(
        repo_id="MSaadAsad/FinBERT-merged-tone-fls",
        filename="finbert_vocab.json"
    )
    
    # Initialize tokenizer
    tokenizer = BertTokenizer(vocab_path)
    
    # Initialize ONNX Runtime session
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_cpu_mem_arena = True
    
    # Create session
    model = onnxruntime.InferenceSession(
        model_path,
        sess_options,
        providers=['CPUExecutionProvider']
    )
    
    return model, tokenizer

def process_batch(model, tokenizer, texts, batch_size):
    """Process a batch of texts."""
    batch_inputs = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': []
    }
    
    for text in texts:
        inputs = tokenizer.encode(text, max_length=512)
        for k in batch_inputs:
            batch_inputs[k].append(inputs[k][0])
    
    # Convert to numpy arrays
    batch_inputs = {k: np.array(v) for k, v in batch_inputs.items()}
    
    # Run inference
    outputs = model.run(None, batch_inputs)
    tone_logits, fls_logits = outputs
    
    # Convert logits to probabilities
    tone_probs = softmax(tone_logits)
    fls_probs = softmax(fls_logits)
    
    return tone_probs, fls_probs

def run_benchmark(num_iterations=10):
    print("Initializing model and tokenizer...")
    model, tokenizer = initialize_model()
    
    # Test sentences
    test_sentences = [
        "We expect strong growth in the next quarter due to our strategic investments.",
        "Revenue declined by 10% compared to last year.",
        "The company maintained stable performance throughout the period.",
        "Our new product launch exceeded expectations significantly.",
        "Market conditions remain challenging but we are optimistic.",
        "We anticipate moderate growth in emerging markets.",
        "Restructuring costs impacted our quarterly results.",
        "The board approved a new share buyback program.",
        "Operating margins improved due to cost optimization.",
        "We successfully completed the acquisition as planned.",
        "Cash flow from operations remained strong.",
        "Research and development expenses increased by 15%.",
        "Customer retention rates exceeded our targets.",
        "We expect to maintain our market leadership position.",
        "The regulatory environment remains uncertain.",
        "Our international expansion plans are on track."
    ]
    
    batch_sizes = [1, 2, 4, 8, 16]
    results = {}
    
    print("\nStarting benchmark...")
    print("-" * 50)
    
    # Warmup phase
    print("\nRunning warmup phase (5 iterations)...")
    for _ in range(5):
        # Warmup sequential
        for text in test_sentences:
            process_batch(model, tokenizer, [text], 1)
        
        # Warmup batch sizes
        for batch_size in batch_sizes:
            batches = [test_sentences[i:i + batch_size] 
                      for i in range(0, len(test_sentences), batch_size)]
            for batch in batches:
                process_batch(model, tokenizer, batch, batch_size)
    print("Warmup complete.")
    
    # Test sequential processing
    print("\nTesting sequential processing...")
    sequential_times = []
    
    for iteration in range(num_iterations):
        if iteration % 10 == 0:
            print(f"Sequential iteration {iteration}/{num_iterations}")
        
        start_time = time.perf_counter()
        for text in test_sentences:
            process_batch(model, tokenizer, [text], 1)
        sequential_times.append(time.perf_counter() - start_time)
    
    results['sequential'] = {
        'mean': np.mean(sequential_times),
        'std': np.std(sequential_times),
        'min': np.min(sequential_times),
        'max': np.max(sequential_times)
    }
    
    # Test different batch sizes
    for batch_size in batch_sizes:
        print(f"\nTesting batch size {batch_size}...")
        batch_times = []
        
        # Create batches
        batches = [test_sentences[i:i + batch_size] 
                  for i in range(0, len(test_sentences), batch_size)]
        
        for iteration in range(num_iterations):
            if iteration % 10 == 0:
                print(f"Batch size {batch_size} iteration {iteration}/{num_iterations}")
            
            start_time = time.perf_counter()
            for batch in batches:
                process_batch(model, tokenizer, batch, batch_size)
            batch_times.append(time.perf_counter() - start_time)
        
        results[f'batch_{batch_size}'] = {
            'mean': np.mean(batch_times),
            'std': np.std(batch_times),
            'min': np.min(batch_times),
            'max': np.max(batch_times)
        }
    
    # Print results
    print("\nBenchmark Results:")
    print("=" * 50)
    print(f"Number of sentences: {len(test_sentences)}")
    print(f"Number of iterations: {num_iterations}")
    print("\nProcessing times (seconds):")
    print("-" * 50)
    
    # Sequential results
    seq_stats = results['sequential']
    print(f"\nSequential processing:")
    print(f"  Mean: {seq_stats['mean']:.4f} ± {seq_stats['std']:.4f}")
    print(f"  Min: {seq_stats['min']:.4f}")
    print(f"  Max: {seq_stats['max']:.4f}")
    
    # Batch results
    for batch_size in batch_sizes:
        stats = results[f'batch_{batch_size}']
        speedup = seq_stats['mean'] / stats['mean']
        print(f"\nBatch size {batch_size}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
        print(f"  Speedup vs sequential: {speedup:.2f}x")
    
    return results

if __name__ == "__main__":
    run_benchmark() 