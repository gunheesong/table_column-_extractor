"""
Granite Vision 3.3 2B wrapper for structured column extraction.

Uses prompt engineering to enforce JSON-only output since Granite Vision
does not support native function calling or JSON mode.
"""

import json
import re

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from .models import ColumnValues


class GraniteVisionExtractor:
    """
    Wrapper for Granite Vision 3.3 2B to extract column values from table images.
    """
    
    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 128,  # Reduced - JSON output only needs ~50 tokens
    ):
        """
        Initialize Granite Vision model.
        
        Args:
            model_path: Local path to the Granite Vision model
            torch_dtype: Torch dtype for model weights
            max_new_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        print("[VLM] Loading model...", flush=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        self.device = next(self.model.parameters()).device
        if torch.cuda.is_available():
            print(f"[VLM] GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
        print(f"[VLM] Model ready on {self.device}", flush=True)
    
    def _resize_image(self, img: Image.Image, max_size: int = 384) -> Image.Image:
        """Resize image to fit within max_size while preserving aspect ratio.
        
        Token count scales with image size:
        - 768px → ~6000 tokens (very slow)
        - 512px → ~2700 tokens (slow)
        - 384px → ~1500 tokens (reasonable)
        """
        if max(img.size) > max_size:
            img = img.copy()
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img
    
    def _generate(self, images: list[Image.Image], prompt: str) -> str:
        """Generate response from model with multiple images."""
        import time
        
        # Resize images to prevent OOM (large PDFs can cause 38GB+ allocations)
        t0 = time.time()
        resized_images = [self._resize_image(img) for img in images]
        print(f"[VLM] Image sizes after resize: {[img.size for img in resized_images]} ({time.time()-t0:.1f}s)", flush=True)
        
        # Build conversation with multiple images
        content = []
        for img in resized_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        
        conversation = [{"role": "user", "content": content}]
        
        t1 = time.time()
        print("[VLM] Applying chat template...", flush=True)
        formatted_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        print(f"[VLM] Chat template done ({time.time()-t1:.1f}s)", flush=True)
        
        t2 = time.time()
        print("[VLM] Processing inputs...", flush=True)
        inputs = self.processor(
            images=resized_images,
            text=formatted_prompt,
            return_tensors="pt"
        )
        print(f"[VLM] Inputs processed ({time.time()-t2:.1f}s)", flush=True)
        print(f"[VLM] Input shapes: {[(k, v.shape) for k, v in inputs.items()]}", flush=True)
        
        # Move to device with correct dtype
        t3 = time.time()
        print("[VLM] Moving to device...", flush=True)
        processed = {}
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                processed[k] = v.to(self.device, dtype=self.torch_dtype)
            else:
                processed[k] = v.to(self.device)
        print(f"[VLM] On device ({time.time()-t3:.1f}s)", flush=True)
        
        if torch.cuda.is_available():
            print(f"[VLM] GPU memory before generate: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
        
        t4 = time.time()
        print(f"[VLM] Starting generation (max {self.max_new_tokens} tokens)...", flush=True)
        
        # Use streamer to show progress
        from transformers import TextStreamer
        streamer = TextStreamer(self.processor, skip_prompt=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **processed,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                streamer=streamer,
            )
        print(f"\n[VLM] Generation done ({time.time()-t4:.1f}s)", flush=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        input_len = processed["input_ids"].shape[1]
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        print(f"[VLM] Response: {response[:100]}..." if len(response) > 100 else f"[VLM] Response: {response}", flush=True)
        return response
    
    def _parse_values(self, response: str) -> list[int]:
        """Parse values from VLM response."""
        # Try direct JSON parse
        try:
            data = json.loads(response.strip())
            if "values" in data:
                return [int(v) for v in data["values"]]
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*"values"\s*:\s*\[[^\]]*\][^{}]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return [int(v) for v in data["values"]]
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Try to extract numbers if JSON parsing fails
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            return [int(float(n)) for n in numbers]
        
        return []
    
    def extract_column_values_single(
        self,
        image: Image.Image,
        table_description: str,
        column_name: str,
    ) -> ColumnValues:
        """
        Extract numeric values from a single image.
        
        Args:
            image: PIL Image containing a table
            table_description: Description of the table to look for
            column_name: Name of the column to extract values from
            
        Returns:
            ColumnValues with list of integers
        """
        prompt = f"""You are analyzing a table image to extract specific data.

TASK:
1. Find the table that contains "{table_description}"
2. In that table, locate the column named "{column_name}" (or similar: y-values, {column_name} values, etc.)
3. Extract ALL numeric values from that column

CRITICAL INSTRUCTIONS:
- The table may have headers like "{column_name}", "y", "{column_name} (units)", etc.
- Extract ONLY numeric values from the target column
- If decimals exist, round to integers
- Return ONLY a JSON object in this exact format:

{{"values": [integer1, integer2, integer3, ...]}}

If no matching table or column is found, return: {{"values": []}}

RESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT."""

        response = self._generate([image], prompt)
        values = self._parse_values(response)
        return ColumnValues(values=values)
    
    def extract_column_values(
        self,
        images: list[Image.Image],
        table_description: str,
        column_name: str,
    ) -> list[ColumnValues]:
        """
        Extract numeric values from each image separately.
        
        Processes images ONE BY ONE to conserve GPU memory.
        
        Args:
            images: List of PIL Images containing tables
            table_description: Description of the table to look for
            column_name: Name of the column to extract values from
            
        Returns:
            List of ColumnValues, one per image
        """
        results = []
        for i, img in enumerate(images):
            print(f"[VLM] Processing image {i+1}/{len(images)}...", flush=True)
            result = self.extract_column_values_single(img, table_description, column_name)
            print(f"[VLM] Found {len(result.values)} values", flush=True)
            results.append(result)
        return results
