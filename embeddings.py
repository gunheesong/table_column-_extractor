"""
Embedding handlers for text and vision models.
"""

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """Wrapper for text embedding model (e.g., Granite embeddings)."""
    
    def __init__(self, model_path: str):
        """
        Initialize the text embedding model.
        
        Args:
            model_path: Local path to the embedding model
        """
        self.model = SentenceTransformer(model_path, trust_remote_code=True)
    
    def embed(self, texts: list[str]) -> torch.Tensor:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of shape (num_texts, embedding_dim)
        """
        return self.model.encode(texts, convert_to_tensor=True)
    
    def embed_single(self, text: str) -> torch.Tensor:
        """Embed a single text string."""
        return self.model.encode([text], convert_to_tensor=True)[0]


class VisionEmbedder:
    """Wrapper for vision embedding model (e.g., CLIP, SigLIP)."""
    
    def __init__(self, model_path: str):
        """
        Initialize the vision embedding model.
        
        Args:
            model_path: Local path to the vision embedding model
        """
        # SentenceTransformers supports CLIP-style models
        self.model = SentenceTransformer(model_path, trust_remote_code=True)
    
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Embed a list of images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Tensor of shape (num_images, embedding_dim)
        """
        return self.model.encode(images, convert_to_tensor=True)
    
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Embed text (for cross-modal similarity with images).
        
        Args:
            text: Text string
            
        Returns:
            Embedding tensor
        """
        return self.model.encode([text], convert_to_tensor=True)[0]


def compute_similarity(query_embedding: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between query and embeddings.
    
    Args:
        query_embedding: Query embedding (1D tensor)
        embeddings: Matrix of embeddings (2D tensor)
        
    Returns:
        Similarity scores for each embedding
    """
    # Normalize
    query_norm = query_embedding / query_embedding.norm()
    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    # Cosine similarity
    return torch.matmul(embeddings_norm, query_norm)

