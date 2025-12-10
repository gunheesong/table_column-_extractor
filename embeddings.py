"""
Embedding handlers for text and vision models.
"""

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor


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
    """Wrapper for Granite Vision Embedding model (ibm-granite/granite-vision-3.3-2b-embedding)."""
    
    def __init__(self, model_path: str):
        """
        Initialize the vision embedding model.
        
        Args:
            model_path: Local path to the Granite Vision Embedding model
        """
        import sys
        
        print(f"[DEBUG] Starting VisionEmbedder init...", flush=True)
        print(f"[DEBUG] GPU available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"[DEBUG] GPU memory before loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
            print(f"[DEBUG] GPU total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)
        
        print(f"[DEBUG] Loading model from: {model_path}", flush=True)
        print(f"[DEBUG] Calling AutoModel.from_pretrained...", flush=True)
        sys.stdout.flush()
        
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            print(f"[DEBUG] Model loaded successfully!", flush=True)
        except Exception as e:
            print(f"[DEBUG] Error during model loading: {e}", flush=True)
            raise
        
        self.model.eval()
        
        print(f"[DEBUG] GPU memory after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
        print(f"[DEBUG] Model dtype: {next(self.model.parameters()).dtype}", flush=True)
        print(f"[DEBUG] Model device: {next(self.model.parameters()).device}", flush=True)
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print(f"[DEBUG] Processor loaded successfully!", flush=True)
    
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Embed a list of images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Tensor of shape (num_images, embedding_dim)
        """
        embeddings = []
        with torch.no_grad():
            for i, img in enumerate(images):
                print(f"[DEBUG] Processing image {i+1}/{len(images)}, size: {img.size}")
                print(f"[DEBUG] GPU memory before processing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
                inputs = self.processor(images=img, return_tensors="pt")
                print(f"[DEBUG] Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
                
                # Move inputs to model's device
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                print(f"[DEBUG] Running forward pass...")
                outputs = self.model(**inputs)
                print(f"[DEBUG] GPU memory after forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
                embeddings.append(outputs.pooler_output.squeeze(0).cpu())
        return torch.stack(embeddings)
    
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Embed text (for cross-modal similarity with images).
        
        Args:
            text: Text string
            
        Returns:
            Embedding tensor
        """
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.pooler_output.squeeze(0).cpu()


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

