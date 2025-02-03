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
    
    num_sentences = len(test_sentences)
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
        elapsed_time = time.perf_counter() - start_time
        sequential_times.append(elapsed_time)
    
    # Calculate throughput for sequential processing
    sequential_throughputs = [num_sentences / t for t in sequential_times]
    results['sequential'] = {
        'mean_time': np.mean(sequential_times),
        'std_time': np.std(sequential_times),
        'min_time': np.min(sequential_times),
        'max_time': np.max(sequential_times),
        'mean_throughput': np.mean(sequential_throughputs),
        'std_throughput': np.std(sequential_throughputs),
        'min_throughput': np.min(sequential_throughputs),
        'max_throughput': np.max(sequential_throughputs)
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
            elapsed_time = time.perf_counter() - start_time
            batch_times.append(elapsed_time)
        
        # Calculate throughput for batch processing
        batch_throughputs = [num_sentences / t for t in batch_times]
        results[f'batch_{batch_size}'] = {
            'mean_time': np.mean(batch_times),
            'std_time': np.std(batch_times),
            'min_time': np.min(batch_times),
            'max_time': np.max(batch_times),
            'mean_throughput': np.mean(batch_throughputs),
            'std_throughput': np.std(batch_throughputs),
            'min_throughput': np.min(batch_throughputs),
            'max_throughput': np.max(batch_throughputs)
        }
    
    # Print results
    print("\nBenchmark Results:")
    print("=" * 50)
    print(f"Number of sentences: {num_sentences}")
    print(f"Number of iterations: {num_iterations}")
    print("\nProcessing statistics:")
    print("-" * 50)
    
    # Sequential results
    seq_stats = results['sequential']
    print(f"\nSequential processing:")
    print(f"  Time: {seq_stats['mean_time']:.4f} ± {seq_stats['std_time']:.4f} seconds")
    print(f"  Throughput: {seq_stats['mean_throughput']:.2f} ± {seq_stats['std_throughput']:.2f} sentences/second")
    print(f"  Min throughput: {seq_stats['min_throughput']:.2f} sentences/second")
    print(f"  Max throughput: {seq_stats['max_throughput']:.2f} sentences/second")
    
    # Batch results
    for batch_size in batch_sizes:
        stats = results[f'batch_{batch_size}']
        throughput_speedup = stats['mean_throughput'] / seq_stats['mean_throughput']
        print(f"\nBatch size {batch_size}:")
        print(f"  Time: {stats['mean_time']:.4f} ± {stats['std_time']:.4f} seconds")
        print(f"  Throughput: {stats['mean_throughput']:.2f} ± {stats['std_throughput']:.2f} sentences/second")
        print(f"  Min throughput: {stats['min_throughput']:.2f} sentences/second")
        print(f"  Max throughput: {stats['max_throughput']:.2f} sentences/second")
        print(f"  Throughput speedup vs sequential: {throughput_speedup:.2f}x")
    
    return results

if __name__ == "__main__":
    run_benchmark() 