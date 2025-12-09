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
        max_new_tokens: int = 512,
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
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    
    def _generate(self, images: list[Image.Image], prompt: str) -> str:
        """Generate response from model with multiple images."""
        # Build conversation with multiple images
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        
        conversation = [{"role": "user", "content": content}]
        
        formatted_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        
        inputs = self.processor(
            images=images,
            text=formatted_prompt,
            return_tensors="pt"
        )
        
        # Move to device with correct dtype
        processed = {}
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                processed[k] = v.to(self.model.device, dtype=self.torch_dtype)
            else:
                processed[k] = v.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **processed,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        input_len = processed["input_ids"].shape[1]
        return self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    def extract_column_values(
        self,
        images: list[Image.Image],
        column_name: str,
    ) -> ColumnValues:
        """
        Extract numeric values from a specified column across table images.
        
        Args:
            images: List of PIL Images containing tables
            column_name: Name of the column to extract values from
            
        Returns:
            ColumnValues with list of integers
        """
        prompt = f"""You are analyzing table images. Your task is to find the column named "{column_name}" and extract all numeric values from that column.

CRITICAL INSTRUCTIONS:
1. Look at ALL provided images
2. Find any table that contains a column called "{column_name}" (exact or close match)
3. Extract ONLY the numeric values from that column
4. Return ONLY a JSON object in this exact format, nothing else:

{{"values": [integer1, integer2, integer3, ...]}}

If the column contains decimal numbers, round them to integers.
If no matching column is found, return: {{"values": []}}

RESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT."""

        response = self._generate(images, prompt)
        
        # Parse JSON from response
        try:
            # Try direct parse first
            data = json.loads(response.strip())
            if "values" in data:
                return ColumnValues(values=[int(v) for v in data["values"]])
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*"values"\s*:\s*\[[^\]]*\][^{}]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return ColumnValues(values=[int(v) for v in data["values"]])
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Try to extract numbers if JSON parsing fails
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            return ColumnValues(values=[int(float(n)) for n in numbers])
        
        return ColumnValues(values=[])

