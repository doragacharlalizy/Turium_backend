"""
Text Chunking Utility
rag_inbox/utils/chunking.py
"""
import re
from typing import List


def estimate_tokens(text: str) -> int:
    """Estimate tokens: ~1 token per 4 characters."""
    return len(text) // 4


def chunk_text_sliding_window(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Token-aware sliding window chunking.
    
    Args:
        text: Text to chunk
        chunk_size: Target tokens per chunk
        overlap: Overlap tokens between chunks
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        
        # If adding this sentence exceeds limit and we have a chunk
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk).strip()
            chunks.append(chunk_text)
            
            # Overlap: keep last sentences
            overlap_tokens = 0
            overlap_sentences = []
            for s in reversed(current_chunk):
                s_tokens = estimate_tokens(s)
                if overlap_tokens + s_tokens <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                else:
                    break
            
            current_chunk = overlap_sentences
            current_tokens = overlap_tokens
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())
    
    return [c for c in chunks if c]