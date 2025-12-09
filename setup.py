from setuptools import setup, find_packages

setup(
    name="table_column_extractor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "transformers>=4.36",
        "accelerate",
        "pymupdf>=1.23",
        "pydantic>=2.0",
        "llama-index-core>=0.10",
        "llama-index-embeddings-huggingface",
        "sentence-transformers>=2.2.0",
        "Pillow",
    ],
    python_requires=">=3.10",
)

