import re
from typing import List

class FastFinancialTokenizer:
    def __init__(self):
        # Fixed-width patterns for lookbehind
        self.pattern = re.compile(
            r'(?<![A-Z]r\.)\s*'           # Mr., Dr.
            r'(?<![A-Z]rs\.)\s*'          # Mrs.
            r'(?<![A-Z]nc\.)\s*'          # Inc.
            r'(?<![A-Z]td\.)\s*'          # Ltd.
            r'(?<![A-Z]o\.)\s*'           # Co.
            r'(?<!\d\.)\s*'               # Numbers
            r'[.!?][\s\n]+(?=[A-Z0-9])'   # Sentence boundary
        )
    
    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        
        # Clean whitespace once
        text = ' '.join(text.split())
        
        # Simple split and clean
        return [s.strip() for s in self.pattern.split(text) if s.strip()]

# Example usage
if __name__ == "__main__":
    tokenizer = FastFinancialTokenizer()
    
    text = """
    Q3 2023 revenue increased by 15%. Our EBITDA margin improved to 23%.
    Mr. Smith projects growth of approx. 10% for Q4.
    Future results may vary. We expect PE ratios to expand in 2024.
    """
    
    sentences = tokenizer.tokenize(text)
    for sentence in sentences:
        print(f"- {sentence}")