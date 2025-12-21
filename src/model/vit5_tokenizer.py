"""ViT5 Tokenizer wrapper for consistency with existing code"""

from transformers import T5Tokenizer
import torch


class ViT5Tokenizer:
    """
    ViT5 tokenizer wrapper
    
    Provides same interface as ViHealthBERTTokenizer for easy replacement
    """
    
    def __init__(
        self, 
        model_name='VietAI/vit5-base',
        max_len=128,
        use_vncorenlp=False,  # Not used for ViT5, kept for compatibility
        vncorenlp_path=None   # Not used for ViT5, kept for compatibility
    ):
        """
        Initialize ViT5 tokenizer
        
        Args:
            model_name: HuggingFace model name (default: 'VietAI/vit5-base')
            max_len: Maximum sequence length
            use_vncorenlp: Not used (kept for compatibility)
            vncorenlp_path: Not used (kept for compatibility)
        """
        print(f"Initializing ViT5 tokenizer: {model_name}")
        
        # Load ViT5 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.model_name = model_name
        
        # Build vocab mapping (for compatibility with existing code)
        self.word_to_idx = {token: idx for token, idx in self.tokenizer.get_vocab().items()}
        self.idx_to_word = {idx: token for token, idx in self.word_to_idx.items()}
        self.vocab_built = True
        
        print(f"✓ ViT5 tokenizer loaded")
        print(f"  Vocab size: {len(self.word_to_idx):,}")
        print(f"  Max length: {max_len}")
    
    def encode(self, text, add_special_tokens=True):
        """
        Encode text to token IDs
        
        Args:
            text: Input Vietnamese text
            add_special_tokens: Whether to add special tokens (ViT5 uses <pad> and </s>)
        
        Returns:
            list: Token IDs (padded to max_len)
        """
        # ViT5 tokenizer automatically handles special tokens
        encoded = self.tokenizer.encode(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors=None  # Return list instead of tensor
        )
        
        return encoded
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to skip special tokens (<pad>, </s>, etc.)
        
        Returns:
            str: Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        
        # Decode using ViT5 tokenizer
        text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        
        return text
    
    def batch_encode(self, texts, add_special_tokens=True):
        """
        Encode multiple texts at once
        
        Args:
            texts: List of input texts
            add_special_tokens: Whether to add special tokens
        
        Returns:
            list: List of token ID lists
        """
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def __len__(self):
        """Return vocabulary size"""
        return len(self.tokenizer)
    
    def __getitem__(self, token):
        """Get token ID"""
        return self.tokenizer.convert_tokens_to_ids(token)
    
    def get_vocab_size(self):
        """Get vocabulary size"""
        return len(self.tokenizer)

