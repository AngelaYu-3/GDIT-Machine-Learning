"""
Initialize Groq AI API Client and Models

This module provides functionality for initializing and configuring the Groq API
clients for use with large language models (LLMs). It handles API key management
and provides a convenient interface for creating both raw Groq clients and 
LangChain-compatible ChatGroq instances.

The module supports multiple API keys with random selection to help with rate limiting
and provides easy access to different Groq models.
"""

from groq import Groq
from langchain_groq import ChatGroq
import random

# List of API keys to use with random selection 
# This helps distribute requests across multiple keys to avoid rate limiting
api_keys = ['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','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',
                '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','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']

def init_groq(model_name = "llama-3.3-70b-versatile"):
    """
    Initialize Groq API client and LangChain compatible model interface.
    
    This function creates both a raw Groq client for direct API access and a
    LangChain-compatible ChatGroq instance configured for the specified model.
    It randomly selects one API key from the available pool to help distribute
    load across API keys.
    
    Args:
        model_name (str, optional): The name of the Groq model to use. 
            Defaults to "llama-3.3-70b-versatile".
            
    Returns:
        tuple: A tuple containing (groq_client, langchain_llm)
            - groq_client: The raw Groq API client for direct API calls
            - langchain_llm: A LangChain-compatible ChatGroq instance
    
    Supported Models:
    ----------------
    Production Models (for production use):
    - distil-whisper-large-v3-en (HuggingFace)
    - gemma2-9b-it (Google): 8,192 tokens context
    - llama-3.3-70b-versatile (Meta): 128k tokens context, 32,768 max completion
    - llama-3.1-8b-instant (Meta): 128k tokens context, 8,192 max completion
    - llama-guard-3-8b (Meta): 8,192 tokens context
    - llama3-70b-8192 (Meta): 8,192 tokens context
    - llama3-8b-8192 (Meta): 8,192 tokens context
    - mixtral-8x7b-32768 (Mistral): 32,768 tokens context
    - whisper-large-v3 (OpenAI): 25 MB max file size
    - whisper-large-v3-turbo (OpenAI): 25 MB max file size
    
    Preview Models (for evaluation only):
    - deepseek-r1-distill-llama-70b-specdec (DeepSeek): 128k tokens context, 16,384 max completion
    - deepseek-r1-distill-llama-70b (DeepSeek): 128k tokens context
    - llama-3.3-70b-specdec (Meta): 8,192 tokens context
    - llama-3.2-1b-preview (Meta): 128k tokens context, 8,192 max completion
    - llama-3.2-3b-preview (Meta): 128k tokens context, 8,192 max completion
    - llama-3.2-11b-vision-preview (Meta): 128k tokens context, 8,192 max completion
    - llama-3.2-90b-vision-preview (Meta): 128k tokens context, 8,192 max completion
    
    Example Usage:
    -------------
    ```python
    # Get both client and LangChain model with default model
    client, llm = init_groq()
    
    # Use with a specific model
    client, llm = init_groq(model_name="mixtral-8x7b-32768")
    
    # Direct client usage
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # LangChain usage
    response = llm.invoke("Tell me about AI")
    ```
    """
    # Create a Groq client with a randomly selected API key
    client = Groq(
        api_key = random.choice(api_keys)
    )

    # Create a LangChain ChatGroq instance with streaming enabled
    llm = ChatGroq(groq_api_key = client.api_key,
                model_name = model_name, streaming=True)
    
    # Return both the raw client and the LangChain model
    return client, llm