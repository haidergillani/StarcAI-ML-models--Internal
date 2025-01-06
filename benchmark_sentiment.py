import time
import torch
from STARC_Cloud_Request import CloudSentimentAnalysis

# Test cases of varying complexity
TEST_CASES = {
    'short': "We expected economic weakness in some emerging markets.",
    
    'medium': """We expected economic weakness in some emerging markets. 
    This turned out to have a significantly greater impact than we had projected.""",
    
    'long': """We had anticipated a slightly shaky economic growth in select emerging markets. 
    This had a greater impact than we were previously expecting. However, while we 
    anticipate a slight dip in quarterly revenue, other items remain broadly aligned 
    with our forecast, which is promising. As we exit a challenging quarter, we are 
    as confident as ever in the fundamental strength of our business. We have always 
    used periods of adversity to re-examine our approach, to take advantage of our 
    culture of flexibility, adaptability, and creativity, and to emerge better as a result."""
}

def run_benchmark(iterations=10):
    # Initialize once
    analyzer = CloudSentimentAnalysis()
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    results = {}
    
    for name, text in TEST_CASES.items():
        print(f"\nTesting {name} text...")
        
        # Debug: Print the merged results
        merged_result = analyzer.get_sentence_sentiments(text)
        print(f"Merged results for {name}: {merged_result}")
        
        # Warmup run with error handling
        try:
            analyzer.cloud_run(text)
        except Exception as e:
            print(f"Error during warmup for {name}: {str(e)}")
            continue
        
        # Timing runs
        times = []
        for i in range(iterations):
            if i % 10 == 0:
                print(f"Running iteration {i}/{iterations}", end='\r')
                
            try:
                start = time.perf_counter()
                analyzer.cloud_run(text)
                end = time.perf_counter()
                times.append(end - start)
            except Exception as e:
                print(f"\nError in iteration {i}: {str(e)}")
                continue
        
        if times:  # Only calculate stats if we have successful runs
            avg_time = sum(times) / len(times)
            results[name] = {
                'avg_time': avg_time,
                'min_time': min(times),
                'max_time': max(times),
                'text_length': len(text),
                'successful_runs': len(times)
            }
        else:
            print(f"No successful runs for {name}")
    
    return results

if __name__ == "__main__":
    print("Starting sentiment analysis benchmark...")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        results = run_benchmark()
        
        print("\nResults:")
        print("-" * 50)
        for name, data in results.items():
            print(f"\n{name.upper()} TEXT:")
            print(f"Characters: {data['text_length']}")
            print(f"Successful runs: {data['successful_runs']}")
            print(f"Average time: {data['avg_time']:.4f} seconds")
            print(f"Min time: {data['min_time']:.4f} seconds")
            print(f"Max time: {data['max_time']:.4f} seconds")
            print(f"Chars/second: {data['text_length']/data['avg_time']:.2f}")
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")