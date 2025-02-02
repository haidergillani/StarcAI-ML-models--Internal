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
    """Initialize the model and tokenizer."""
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
        
        # Initialize ONNX Runtime session
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL  # For consistent performance
        sess_options.enable_cpu_mem_arena = True  # Enable memory arena for better caching
        
        # Create session
        _model = onnxruntime.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )

def analyze_batch(texts, batch_size=16, max_length=512):
    """Process a batch of texts."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Process each text in batch
        batch_inputs = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': []
        }
        
        for text in batch:
            inputs = _tokenizer.encode(text, max_length)
            for k in batch_inputs:
                batch_inputs[k].append(inputs[k][0])
        
        # Convert to numpy arrays
        batch_inputs = {k: np.array(v) for k, v in batch_inputs.items()}
        
        # Run inference
        outputs = _model.run(None, batch_inputs)
        tone_logits, fls_logits = outputs
        
        # Convert logits to probabilities
        tone_probs = softmax(tone_logits)
        fls_probs = softmax(fls_logits)
        
        for tone_prob, fls_prob in zip(tone_probs, fls_probs):
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
    print(f"Environment variables available: {list(os.environ.keys())}")
    print(f"Expected API key exists: {bool(expected_api_key)}")
    
    if not expected_api_key:
        raise ValueError("Google Cloud API key not found in environment variables. Please check configuration.")
    
    request_api_key = request.args.get('apikey')
    print(f"Received API key exists: {bool(request_api_key)}")
    
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
            results = analyze_batch(texts)
            return (json.dumps(results[0]), 200, headers)
        elif 'texts' in request_json:
            texts = request_json['texts']
            results = analyze_batch(texts)
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
    print(analyze_batch(texts)) 