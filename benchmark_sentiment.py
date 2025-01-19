import time
import torch
from STARC_Cloud_Request import CloudSentimentAnalysis

TEST_CASES = {
    'short': "We expected economic weakness in some emerging markets.",
    
    'medium': """We expected economic weakness in some emerging markets. 
    This turned out to have a significantly greater impact than we had projected.""",
    
    'long': """We had anticipated a slightly shaky economic growth in select emerging markets. 
    This had a greater impact than we were previously expecting. However, while we 
    anticipate a slight dip in quarterly revenue, other items remain broadly aligned 
    with our forecast, which is promising. As we exit a challenging quarter, we are 
    as confident as ever in the fundamental strength of our business."""
}

def run_benchmark(iterations=10):
    analyzer = CloudSentimentAnalysis()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    results = {}
    
    for name, text in TEST_CASES.items():
        print(f"\nTesting {name} text...")
        timings = {'tone': [], 'fls': [], 'merge': [], 'total': []}
        
        # Warmup run
        analyzer.cloud_run(text)
        
        for i in range(iterations):
            if i % 10 == 0:
                print(f"Running iteration {i}/{iterations}", end='\r')
            
            try:
                # Time each component
                start_total = time.perf_counter()
                
                # Tone Model
                start = time.perf_counter()
                tone_results = analyzer.tone_model.get_sentiments(text)
                timings['tone'].append(time.perf_counter() - start)
                
                # FLS Model
                start = time.perf_counter()
                fls_results = analyzer.FLS_model.get_sentimentsFLS(text)
                timings['fls'].append(time.perf_counter() - start)
                
                # Merge
                start = time.perf_counter()
                merged = analyzer.merge_data(tone_results, fls_results)
                timings['merge'].append(time.perf_counter() - start)
                
                # Total time
                timings['total'].append(time.perf_counter() - start_total)
                
            except Exception as e:
                print(f"\nError in iteration {i}: {str(e)}")
                continue
        
        # Calculate statistics
        results[name] = {
            'text_length': len(text),
            'components': {
                component: {
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
                for component, times in timings.items()
            }
        }
    
    return results

if __name__ == "__main__":
    print("Starting sentiment analysis benchmark...")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        results = run_benchmark()
        
        print("\nDetailed Results:")
        print("-" * 50)
        for name, data in results.items():
            print(f"\n{name.upper()} TEXT ({data['text_length']} chars):")
            total_time = data['components']['total']['avg_time']
            
            for component, timing in data['components'].items():
                if component != 'total':
                    percentage = (timing['avg_time'] / total_time) * 100
                    print(f"{component:<6}: {timing['avg_time']:.4f}s ({percentage:.1f}%)")
                    print(f"       min: {timing['min_time']:.4f}s, max: {timing['max_time']:.4f}s")
            
            print(f"Total:  {total_time:.4f}s")
            print(f"Chars/second: {data['text_length']/total_time:.2f}")
            
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")