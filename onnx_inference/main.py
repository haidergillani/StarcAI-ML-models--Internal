import json
import os
from dotenv import load_dotenv
import functions_framework
from huggingface_hub import hf_hub_download
import onnxruntime
import numpy as np
from tokenizer import BertTokenizer

# Load environment variables
load_dotenv()

# Global variables to cache model and tokenizer
_model = None
_tokenizer = None

# Define labels
tone_labels = ['Neutral', 'Positive', 'Negative']
fls_labels = ['Not FLS', 'Non-specific FLS', 'Specific FLS']

def initialize():
    """Initialize the model and tokenizer with optimized settings."""
    global _model, _tokenizer
    
    if _model is None:
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
        _tokenizer = BertTokenizer(vocab_path)
        
        # Initialize ONNX Runtime session with optimized settings
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Get available providers
        available_providers = onnxruntime.get_available_providers()
        
        # Choose best available provider
        if 'CUDAExecutionProvider' in available_providers:
            provider = 'CUDAExecutionProvider'
            # CUDA-specific optimizations
            sess_options.enable_cuda_graph = True
            sess_options.cuda_mem_limit = 2 * 1024 * 1024 * 1024  # 2GB CUDA memory limit
        else:
            provider = 'CPUExecutionProvider'
            # CPU-specific optimizations
            num_threads = min(8, os.cpu_count() or 4)  # Use up to 8 threads
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        
        # Memory optimizations
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        
        # Create session
        _model = onnxruntime.InferenceSession(
            model_path,
            sess_options,
            providers=[provider]
        )

def process_batch(texts, batch_size=2, max_length=512):
    """Process a batch of texts with optimized memory handling."""
    results = []
    
    # Add padding text if we have odd number of texts
    if len(texts) % 2 != 0:
        texts = texts + [""]  # Add empty text for padding
    
    # Process texts in batches of 2
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Pre-allocate numpy arrays for the batch
        input_ids = np.zeros((batch_size, max_length), dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_length), dtype=np.int64)
        token_type_ids = np.zeros((batch_size, max_length), dtype=np.int64)
        
        # Tokenize all texts at once and fill pre-allocated arrays
        for j, text in enumerate(batch):
            encoded = _tokenizer.encode(text, max_length)
            input_ids[j] = encoded['input_ids'][0]
            attention_mask[j] = encoded['attention_mask'][0]
            token_type_ids[j] = encoded['token_type_ids'][0]
        
        # Create batch inputs once
        batch_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
        
        # Run inference
        outputs = _model.run(None, batch_inputs)
        tone_logits, fls_logits = outputs
        
        # Convert logits to probabilities
        tone_probs = softmax(tone_logits)
        fls_probs = softmax(fls_logits)
        
        # Format results, but skip the padding result
        for k, (tone_prob, fls_prob) in enumerate(zip(tone_probs, fls_probs)):
            if i + k < len(texts) - (1 if len(texts) % 2 != 0 else 0):  # Skip padding result
                results.append({
                    'tone': {
                        label: float(prob)
                        for label, prob in zip(tone_labels, tone_prob)
                    },
                    'fls': {
                        label: float(prob)
                        for label, prob in zip(fls_labels, fls_prob)
                    }
                })
    
    return results

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def validate_api_key(request):
    """Validate the API key from request."""
    expected_api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
    
    if not expected_api_key:
        raise ValueError("Google Cloud API key not found in environment variables. Please check configuration.")
    
    request_api_key = request.args.get('apikey')
    
    if not request_api_key:
        raise ValueError("No API key provided in request")
    
    if request_api_key != expected_api_key:
        raise ValueError("Invalid API key provided")
    
    return True

@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function for model inference."""
    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Set CORS headers for the main request
    headers = {'Access-Control-Allow-Origin': '*'}
    
    try:
        # Validate API key
        if not validate_api_key(request):
            return ('Invalid API key', 403, headers)
            
        # Initialize if needed
        if _model is None:
            initialize()
        
        # Get request data
        request_json = request.get_json(silent=True)
        if not request_json:
            return ('No data provided', 400, headers)
        
        # Handle both single text and batch of texts
        if 'text' in request_json:
            texts = [request_json['text']]
            results = process_batch(texts)
            return (json.dumps(results[0]), 200, headers)
        elif 'texts' in request_json:
            texts = request_json['texts']
            results = process_batch(texts)
            return (json.dumps(results), 200, headers)
        else:
            return ('No text or texts provided', 400, headers)
        
    except Exception as e:
        return (f'Error processing request: {str(e)}', 500, headers)

if __name__ == "__main__":
    # For local testing
    load_dotenv()  # Load environment variables for local testing
    initialize()
    texts = [
        "We expect strong growth in the next quarter due to our strategic investments.",
        "Revenue declined by 10% compared to last year."
    ]
    print(process_batch(texts)) 