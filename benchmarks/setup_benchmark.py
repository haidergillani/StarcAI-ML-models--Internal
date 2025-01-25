import time
import subprocess
import sys
from pathlib import Path
import os
import pkg_resources
from functools import wraps

# Get the absolute path to the requirements.txt file
REPO_ROOT = Path(__file__).parent.parent
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

# Add directories to Python path
MODELS_PATH = REPO_ROOT / "models"
UTILS_PATH = REPO_ROOT / "utils"
sys.path.append(str(MODELS_PATH))
sys.path.append(str(UTILS_PATH))

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

class SetupBenchmark:
    def __init__(self):
        self.timings = {}
        self.detailed_timings = {}
        
    def get_installed_packages(self):
        return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    @time_operation("pip_force_reinstall")
    def force_reinstall_requirements(self):
        print("Force reinstalling requirements...")
        before_packages = self.get_installed_packages()
        
        with open(REQUIREMENTS_PATH, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        total_start = time.perf_counter()
        for req in requirements:
            pkg_name = req.split('==')[0] if '==' in req else req
            print(f"Reinstalling {pkg_name}...")
            start = time.perf_counter()
            try:
                subprocess.check_call([
                    sys.executable, 
                    "-m", 
                    "pip", 
                    "install", 
                    "--force-reinstall",
                    "--no-deps",  # Don't reinstall dependencies
                    req
                ])
                end = time.perf_counter()
                self.detailed_timings[f"reinstall_{pkg_name}"] = end - start
            except subprocess.CalledProcessError as e:
                print(f"Failed to reinstall {pkg_name}: {str(e)}")
        
        after_packages = self.get_installed_packages()
        return before_packages, after_packages

    @time_operation("model_download_and_init")
    def initialize_models(self):
        print("Initializing models...")
        start = time.perf_counter()
        from Tone_Model import ToneModel
        tone_init_time = time.perf_counter() - start
        self.detailed_timings['tone_model_init'] = tone_init_time
        
        start = time.perf_counter()
        from FLS_Model import FLSModel
        fls_init_time = time.perf_counter() - start
        self.detailed_timings['fls_model_init'] = fls_init_time
        
        print("Loading models into memory...")
        start = time.perf_counter()
        tone_model = ToneModel()
        tone_load_time = time.perf_counter() - start
        self.detailed_timings['tone_model_load'] = tone_load_time
        
        start = time.perf_counter()
        fls_model = FLSModel()
        fls_load_time = time.perf_counter() - start
        self.detailed_timings['fls_model_load'] = fls_load_time
        
        return tone_model, fls_model

    @time_operation("tokenizer_setup")
    def setup_tokenizer(self):
        print("Setting up tokenizer...")
        from transformers import BertTokenizer
        start = time.perf_counter()
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.detailed_timings['tokenizer_download'] = time.perf_counter() - start
        return tokenizer

    @time_operation("nltk_install")
    def install_nltk(self):
        print("Installing NLTK...")
        start = time.perf_counter()
        try:
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "nltk"
            ])
            self.detailed_timings['nltk_install'] = time.perf_counter() - start
        except subprocess.CalledProcessError as e:
            print(f"Failed to install NLTK: {str(e)}")

    @time_operation("nltk_download")
    def download_nltk_data(self):
        print("Downloading NLTK data...")
        import nltk
        
        start = time.perf_counter()
        nltk.download('punkt')
        self.detailed_timings['nltk_punkt_download'] = time.perf_counter() - start
        
        start = time.perf_counter()
        nltk.download('averaged_perceptron_tagger')
        self.detailed_timings['nltk_tagger_download'] = time.perf_counter() - start

    @time_operation("sample_inference")
    def run_sample_inference(self, tone_model, fls_model):
        print("Running sample inference...")
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
            # Time tone model inference
            start = time.perf_counter()
            tone_results = tone_model.get_sentiments(text)
            self.detailed_timings[f'tone_inference_{length}'] = time.perf_counter() - start
            
            # Time FLS model inference
            start = time.perf_counter()
            fls_results = fls_model.get_sentimentsFLS(text)
            self.detailed_timings[f'fls_inference_{length}'] = time.perf_counter() - start

    @time_operation("cleanup")
    def cleanup_environment(self):
        print("Cleaning up environment...")
        
        # Clear pip cache
        print("Clearing pip cache...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "cache", "purge"])
        except:
            print("No pip cache to clear")
            
        # Clear NLTK data
        print("Clearing NLTK data...")
        import shutil
        nltk_data_path = os.path.expanduser('~/nltk_data')
        if os.path.exists(nltk_data_path):
            shutil.rmtree(nltk_data_path)
            
        # Clear HuggingFace cache
        print("Clearing HuggingFace cache...")
        cache_path = os.path.expanduser('~/.cache/huggingface')
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            
        # Clear PyTorch hub cache
        print("Clearing PyTorch hub cache...")
        torch_hub_path = os.path.expanduser('~/.cache/torch')
        if os.path.exists(torch_hub_path):
            shutil.rmtree(torch_hub_path)
            
        # Uninstall packages
        print("Uninstalling packages...")
        packages_to_uninstall = ['nltk', 'torch', 'transformers', 'numpy']
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
        print("Starting full benchmark...")
        # Move torch import after requirements installation
        print(f"Python version: {sys.version}")
        
        try:
            # Clean everything first
            print("\nStep 1: Cleaning Environment")
            self.cleanup_environment()
            
            # First install all requirements
            print("\nStep 2: Installing Requirements")
            before_pkgs, after_pkgs = self.force_reinstall_requirements()
            
            # Now we can import torch since it's installed
            import torch
            print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            
            # Install NLTK separately
            print("\nStep 3: Installing NLTK")
            self.install_nltk()
            
            # Now download NLTK data
            print("\nStep 4: Downloading NLTK Data")
            self.download_nltk_data()
            
            print("\nStep 5: Initializing Models")
            tone_model, fls_model = self.initialize_models()
            
            print("\nStep 6: Setting up Tokenizer")
            self.setup_tokenizer()
            
            print("\nStep 7: Running Sample Inference")
            self.run_sample_inference(tone_model, fls_model)
            
            # Print results
            print("\nDetailed Benchmark Results:")
            print("-" * 60)
            
            # Group and print results by category
            categories = {
                'Environment Cleanup': ['cleanup'],
                'Package Installation': [k for k in self.detailed_timings.keys() if k.startswith('reinstall_')],
                'NLTK Setup': ['nltk_install', 'nltk_punkt_download', 'nltk_tagger_download'],
                'Model Setup': ['tone_model_init', 'fls_model_init', 'tone_model_load', 'fls_model_load', 'tokenizer_download'],
                'Inference (Short Text)': ['tone_inference_short', 'fls_inference_short'],
                'Inference (Medium Text)': ['tone_inference_medium', 'fls_inference_medium'],
                'Inference (Long Text)': ['tone_inference_long', 'fls_inference_long']
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
    benchmark = SetupBenchmark()
    benchmark.run_full_benchmark() 