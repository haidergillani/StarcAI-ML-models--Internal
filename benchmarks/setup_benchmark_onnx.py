import time
import subprocess
import sys
import os
import pkg_resources
from functools import wraps
import shutil
import json
import numpy as np

def time_operation(operation_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.perf_counter()
            result = func(self, *args, **kwargs)
            end_time = time.perf_counter()
            self.timings[operation_name] = end_time - start_time
            return result
        return wrapper
    return decorator

class SetupBenchmarkONNX:
    def __init__(self):
        self.timings = {}
        self.detailed_timings = {}
        
    def get_installed_packages(self):
        return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    @time_operation("pip_force_reinstall")
    def force_reinstall_requirements(self):
        print("Force reinstalling requirements...")
        before_packages = self.get_installed_packages()
        
        requirements = [
            'onnxruntime==1.16.3',
            'numpy==1.24.3',
            'huggingface-hub==0.19.4',
            'onnx==1.15.0'
        ]
        
        total_start = time.perf_counter()
        for req in requirements:
            pkg_name = req.split('==')[0]
            print(f"Reinstalling {pkg_name}...")
            start = time.perf_counter()
            try:
                subprocess.check_call([
                    sys.executable, 
                    "-m", 
                    "pip", 
                    "install", 
                    "--force-reinstall",
                    "--no-deps",
                    req
                ])
                end = time.perf_counter()
                self.detailed_timings[f"reinstall_{pkg_name}"] = end - start
            except subprocess.CalledProcessError as e:
                print(f"Failed to reinstall {pkg_name}: {str(e)}")
        
        after_packages = self.get_installed_packages()
        return before_packages, after_packages

    @time_operation("model_download")
    def download_model(self):
        print("Downloading model...")
        from huggingface_hub import hf_hub_download
        import onnx
        
        start = time.perf_counter()
        model_path = hf_hub_download(
            repo_id="MSaadAsad/FinBERT-merged-tone-fls",
            filename="finbert_6layers_quantized_compat.onnx"
        )
        
        # Print detailed model info
        print(f"Downloaded model to: {model_path}")
        print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        # Load and print detailed model metadata
        try:
            model = onnx.load(model_path)
            print("\nDetailed Model Metadata:")
            print(f"IR version: {model.ir_version}")
            print(f"Producer name: {model.producer_name}")
            print(f"Producer version: {model.producer_version}")
            print(f"Domain: {model.domain}")
            print(f"Model version: {model.model_version}")
            print("\nAll Opset Imports:")
            for opset in model.opset_import:
                print(f"Domain: {opset.domain}, Version: {opset.version}")
            print("\nOperator Domains Used:")
            ops = set()
            for node in model.graph.node:
                ops.add(node.domain if node.domain else "ai.onnx")
            print("Domains:", ops)
        except Exception as e:
            print(f"Failed to load model metadata: {str(e)}")
        
        self.detailed_timings['model_download'] = time.perf_counter() - start
        return model_path

    @time_operation("tokenizer_setup")
    def setup_tokenizer(self):
        print("Setting up tokenizer...")
        from huggingface_hub import hf_hub_download
        
        start = time.perf_counter()
        vocab_path = hf_hub_download(
            repo_id="MSaadAsad/FinBERT-merged-tone-fls",
            filename="finbert_vocab.json"
        )
        self.detailed_timings['vocab_download'] = time.perf_counter() - start
        
        start = time.perf_counter()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.detailed_timings['vocab_load'] = time.perf_counter() - start
        
        return vocab_path

    @time_operation("model_init")
    def initialize_model(self, model_path):
        print("Initializing ONNX model...")
        import onnxruntime
        import platform
        import os
        
        # Set environment variables to bypass opset version checks and enable more detailed logging
        os.environ['ORT_DISABLE_STRICT_OPSET_CHECKING'] = '1'
        os.environ['ORT_DISABLE_FALLBACK'] = '0'
        os.environ['ORT_LOG_LEVEL'] = '0'  # Verbose logging
        os.environ['ONNXRUNTIME_ENABLE_EXTENDED_OPERATORS'] = '1'
        os.environ['ORT_DISABLE_ALL_OPSET_VALIDATION'] = '1'  # Try to completely disable opset validation
        
        # Print detailed environment info
        print("\nEnvironment Details:")
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Python version: {platform.python_version()}")
        print(f"ONNX Runtime version: {onnxruntime.__version__}")
        print(f"Available providers: {onnxruntime.get_available_providers()}")
        
        start = time.perf_counter()
        
        # Simplified session options - focus on basic CPU execution
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        
        # Try to disable opset validation in session options
        sess_options.add_session_config_entry('session.disable_strict_opset_checking', '1')
        sess_options.add_session_config_entry('session.use_ort_model_bytes_directly', '1')
        
        try:
            print("\nAttempting to create inference session...")
            session = onnxruntime.InferenceSession(
                model_path,
                sess_options,
                providers=['CPUExecutionProvider']
            )
            
            print("Successfully created inference session")
            print("Model inputs:", session.get_inputs())
            print("Model outputs:", session.get_outputs())
            
            self.detailed_timings['model_init'] = time.perf_counter() - start
            return session
            
        except Exception as e:
            print(f"\nFailed to initialize model: {str(e)}")
            print(f"Model path: {model_path}")
            if os.path.exists(model_path):
                print(f"Model file size: {os.path.getsize(model_path)} bytes")
            else:
                print("Model file does not exist!")
            raise

    @time_operation("sample_inference")
    def run_sample_inference(self, session, vocab_path):
        print("Running sample inference...")
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from onnx_inference.tokenizer import BertTokenizer
        
        tokenizer = BertTokenizer(vocab_path)
        
        sample_texts = {
            'short': "We expected economic weakness in some emerging markets.",
            'medium': """We expected economic weakness in some emerging markets. 
            This turned out to have a significantly greater impact than we had projected.""",
            'long': """We had anticipated a slightly shaky economic growth in select emerging markets. 
            This had a greater impact than we were previously expecting. However, while we 
            anticipate a slight dip in quarterly revenue, other items remain broadly aligned 
            with our forecast, which is promising. As we exit a challenging quarter, we are 
            as confident as ever in the fundamental strength of our business."""
        }
        
        for length, text in sample_texts.items():
            # Time tokenization
            start = time.perf_counter()
            inputs = tokenizer.encode(text, max_length=512)
            self.detailed_timings[f'tokenization_{length}'] = time.perf_counter() - start
            
            # Time inference
            start = time.perf_counter()
            outputs = session.run(None, inputs)
            self.detailed_timings[f'inference_{length}'] = time.perf_counter() - start
            
            # Time post-processing
            start = time.perf_counter()
            tone_logits, fls_logits = outputs
            def softmax(x):
                exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return exp_x / np.sum(exp_x, axis=1, keepdims=True)
            tone_probs = softmax(tone_logits)[0]
            fls_probs = softmax(fls_logits)[0]
            self.detailed_timings[f'postprocess_{length}'] = time.perf_counter() - start

    @time_operation("batch_inference")
    def run_batch_inference(self, session, vocab_path, batch_size=32):
        print("Running batch inference...")
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from onnx_inference.tokenizer import BertTokenizer
        
        tokenizer = BertTokenizer(vocab_path)
        
        # Create a batch of texts
        texts = [
            "Revenue declined by 10% compared to last year.",
            "We project a 15% increase in market share.",
            "The company maintained stable performance.",
            "We expect strong growth in the next quarter."
        ] * (batch_size // 4)  # Repeat to fill batch
        
        # Time batch tokenization
        start = time.perf_counter()
        batch_inputs = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': []
        }
        for text in texts:
            inputs = tokenizer.encode(text, max_length=512)
            for k in batch_inputs:
                batch_inputs[k].append(inputs[k][0])
        batch_inputs = {k: np.array(v) for k, v in batch_inputs.items()}
        self.detailed_timings['batch_tokenization'] = time.perf_counter() - start
        
        # Time batch inference
        start = time.perf_counter()
        outputs = session.run(None, batch_inputs)
        self.detailed_timings['batch_inference'] = time.perf_counter() - start
        
        # Time batch post-processing
        start = time.perf_counter()
        tone_logits, fls_logits = outputs
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        tone_probs = softmax(tone_logits)
        fls_probs = softmax(fls_logits)
        self.detailed_timings['batch_postprocess'] = time.perf_counter() - start

    @time_operation("cleanup")
    def cleanup_environment(self):
        print("Cleaning up environment...")
        
        # Clear pip cache
        print("Clearing pip cache...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "cache", "purge"])
        except:
            print("No pip cache to clear")
            
        # Clear HuggingFace cache
        print("Clearing HuggingFace cache...")
        cache_path = os.path.expanduser('~/.cache/huggingface')
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            
        # Uninstall packages
        print("Uninstalling packages...")
        packages_to_uninstall = ['onnxruntime', 'numpy', 'huggingface-hub']
        for pkg in packages_to_uninstall:
            try:
                subprocess.check_call([
                    sys.executable,
                    "-m",
                    "pip",
                    "uninstall",
                    "-y",
                    pkg
                ])
            except:
                print(f"Failed to uninstall {pkg}")

    def run_full_benchmark(self):
        print("Starting full ONNX benchmark...")
        print(f"Python version: {sys.version}")
        
        try:
            # Clean everything first
            print("\nStep 1: Cleaning Environment")
            self.cleanup_environment()
            
            # First install all requirements
            print("\nStep 2: Installing Requirements")
            before_pkgs, after_pkgs = self.force_reinstall_requirements()
            
            # Download model
            print("\nStep 3: Downloading Model")
            model_path = self.download_model()
            
            # Setup tokenizer
            print("\nStep 4: Setting up Tokenizer")
            vocab_path = self.setup_tokenizer()
            
            # Initialize model
            print("\nStep 5: Initializing Model")
            session = self.initialize_model(model_path)
            
            # Run sample inference
            print("\nStep 6: Running Sample Inference")
            self.run_sample_inference(session, vocab_path)
            
            # Run batch inference
            print("\nStep 7: Running Batch Inference")
            self.run_batch_inference(session, vocab_path)
            
            # Print results
            print("\nDetailed Benchmark Results:")
            print("-" * 60)
            
            # Group and print results by category
            categories = {
                'Environment Setup': ['cleanup'],
                'Package Installation': [k for k in self.detailed_timings.keys() if k.startswith('reinstall_')],
                'Model Setup': ['model_download', 'vocab_download', 'vocab_load', 'model_init'],
                'Single Inference': [
                    'tokenization_short', 'inference_short', 'postprocess_short',
                    'tokenization_medium', 'inference_medium', 'postprocess_medium',
                    'tokenization_long', 'inference_long', 'postprocess_long'
                ],
                'Batch Inference': ['batch_tokenization', 'batch_inference', 'batch_postprocess']
            }
            
            for category, operations in categories.items():
                print(f"\n{category}:")
                print("-" * 40)
                category_total = 0
                for op in operations:
                    if op in self.detailed_timings:
                        duration = self.detailed_timings[op]
                        category_total += duration
                        print(f"{op:<35}: {duration:.2f} seconds")
                print(f"{'Category Total':<35}: {category_total:.2f} seconds")
            
            print("\nOverall Timings:")
            print("-" * 40)
            for operation, duration in self.timings.items():
                print(f"{operation:<35}: {duration:.2f} seconds")
            
        except Exception as e:
            print(f"Benchmark failed: {str(e)}")
            raise

if __name__ == "__main__":
    benchmark = SetupBenchmarkONNX()
    benchmark.run_full_benchmark() 