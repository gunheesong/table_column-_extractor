"""Debug script to find which model is causing the 38GB allocation."""

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Current allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("\n" + "="*60)
print("STEP 1: Testing Text Embedding Model")
print("="*60)

try:
    from sentence_transformers import SentenceTransformer
    
    # Replace this with your actual text embedding path
    TEXT_EMB_PATH = "./textembedding"  # CHANGE THIS
    
    print(f"Loading from: {TEXT_EMB_PATH}")
    print("Loading SentenceTransformer...", flush=True)
    
    model = SentenceTransformer(TEXT_EMB_PATH, trust_remote_code=True)
    
    print(f"Text model loaded successfully!")
    if torch.cuda.is_available():
        print(f"GPU memory after text model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Test it
    emb = model.encode(["test"], convert_to_tensor=True)
    print(f"Text embedding shape: {emb.shape}")
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    print("✓ Text embedding OK\n")
    
except Exception as e:
    print(f"✗ Text embedding FAILED: {e}\n")

print("="*60)
print("STEP 2: Testing VLM Model")
print("="*60)

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    
    # Replace this with your actual VLM path
    VLM_PATH = "./vlm"  # CHANGE THIS
    
    print(f"Loading from: {VLM_PATH}")
    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(VLM_PATH)
    print("Processor loaded!", flush=True)
    
    print("Loading VLM to CPU (no device_map)...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        VLM_PATH,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    print("VLM loaded to CPU!", flush=True)
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    if torch.cuda.is_available():
        print(f"GPU memory before .cuda(): {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
        print("Moving to GPU...", flush=True)
        model = model.cuda()
        print(f"GPU memory after .cuda(): {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    
    print("✓ VLM OK\n")
    
except Exception as e:
    print(f"✗ VLM FAILED: {e}\n")
    import traceback
    traceback.print_exc()

print("="*60)
print("DEBUG COMPLETE")
print("="*60)

