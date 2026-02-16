"""
LLM Client - Handles all LLM operations (OpenAI, Claude, Ollama, etc.)
"""
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# CHOOSE YOUR LLM HERE - Uncomment the one you want to use
# ============================================================================

USE_OPENAI = False
USE_CLAUDE = False
USE_OLLAMA = True         # ← SET TO TRUE FOR OLLAMA (FREE!)
USE_GROQ = False

# ============================================================================
# Ollama Implementation (Completely FREE & Local) - ACTIVE
# ============================================================================

if USE_OLLAMA:
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')
    
    # Create OpenAI client pointing to Ollama (OpenAI-compatible API)
    ollama_client = OpenAI(
        api_key="ollama",  # Dummy key for local Ollama
        base_url=f"{OLLAMA_URL}/v1"  # Use OpenAI-compatible endpoint
    )
    
    # Test connection
    try:
        import requests
        response = requests.get(f'{OLLAMA_URL}/api/tags', timeout=2)
        if response.status_code == 200:
            logger.info(f"✅ Ollama connected at {OLLAMA_URL}")
        else:
            logger.warning(f"⚠️  Ollama responded with status {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️  Ollama not responding at {OLLAMA_URL}")
        logger.warning(f"   Make sure to run: ollama serve")
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed(text: str) -> List[float]:
        """Generate embedding using HuggingFace (free, local)."""
        try:
            embeddings = embedding_model.encode(text)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise
    
    def embed_batch(texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            embeddings = embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Batch embedding error: {str(e)}")
            raise
    
    def complete(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate completion using Ollama (via OpenAI-compatible API)."""
        try:
            response = ollama_client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Ollama completion error: {str(e)}")
            raise

# ============================================================================
# Claude Implementation (Paid, but excellent)
# ============================================================================

elif USE_CLAUDE:
    from anthropic import Anthropic
    from sentence_transformers import SentenceTransformer
    
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")
    
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed(text: str) -> List[float]:
        """Generate embedding using HuggingFace."""
        try:
            embeddings = embedding_model.encode(text)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise
    
    def embed_batch(texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            embeddings = embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Batch embedding error: {str(e)}")
            raise
    
    def complete(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate completion using Claude."""
        try:
            message = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Claude completion error: {str(e)}")
            raise

# ============================================================================
# OpenAI Implementation (Paid)
# ============================================================================

elif USE_OPENAI:
    from openai import OpenAI
    
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    def embed(text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise
    
    def embed_batch(texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            embeddings = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in embeddings]
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {str(e)}")
            raise
    
    def complete(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate completion using OpenAI."""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI completion error: {str(e)}")
            raise

# ============================================================================
# Groq Implementation (Free with limits)
# ============================================================================

elif USE_GROQ:
    from groq import Groq
    from sentence_transformers import SentenceTransformer
    
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment")
    
    groq_client = Groq(api_key=GROQ_API_KEY)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed(text: str) -> List[float]:
        """Generate embedding using HuggingFace."""
        try:
            embeddings = embedding_model.encode(text)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise
    
    def embed_batch(texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            embeddings = embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Batch embedding error: {str(e)}")
            raise
    
    def complete(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate completion using Groq."""
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="mixtral-8x7b-32768",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq completion error: {str(e)}")
            raise

# ============================================================================
# Default: Fallback to ensure functions exist
# ============================================================================

else:
    def embed(text: str) -> List[float]:
        raise ValueError("No LLM provider configured")
    
    def embed_batch(texts: List[str]) -> List[List[float]]:
        raise ValueError("No LLM provider configured")
    
    def complete(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        raise ValueError("No LLM provider configured")

# ============================================================================
# Main Client Class
# ============================================================================

class LLMClient:
    """Unified LLM Client interface."""
    
    def __init__(self):
        pass
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding."""
        return embed(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings."""
        return embed_batch(texts)
    
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate completion."""
        return complete(system_prompt, user_prompt, temperature, max_tokens)

# ============================================================================
# Singleton Instance
# ============================================================================

_llm_client_instance = None

def get_llm_client() -> LLMClient:
    """Get or create LLM client singleton."""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance

# ============================================================================
# Function Exports (for backward compatibility)
# ============================================================================

__all__ = [
    'get_llm_client',
    'embed',
    'embed_batch',
    'complete',
    'LLMClient',
]