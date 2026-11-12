
# Neural RAG & Foundation Model Research Experiment

## Overview
A research-oriented Retrieval-Augmented Generation (RAG) system that explores how document chunking and embedding configurations affect information retrieval and answer generation.

## Technology Stack
- Python 3.11
- PyTorch
- JAX
- Hugging Face Transformers
- Sentence Transformers (MiniLM)
- LangChain
- ChromaDB
- FastAPI
- NumPy and Pandas

## Architecture
1. Dataset preparation
2. Document preprocessing
3. Dynamic text chunking
4. MiniLM embedding generation
5. ChromaDB vector storage
6. Similarity-based retrieval
7. FLAN-T5 answer generation
8. Experimental evaluation

## Experiments
15 configurations are defined to compare:
- Chunk sizes
- Chunk overlap
- Retrieval top-k values
- Embedding generation time

## Evaluation
- Retrieval recall proxy
- Mean Reciprocal Rank (MRR)
- Retrieval latency
- Perplexity evaluation

## Hardware
CPU-based execution using Intel Iris Xe integrated graphics.

## Project Status
Core RAG implementation and experimental evaluation are under development.
