
import os
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

texts = [
    "Machine learning enables computers to learn from data.",
    "Deep learning uses neural networks.",
    "RAG combines retrieval with text generation.",
    "Python is a popular programming language."
]

# MiniLM embeddings
mini_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

start = time.perf_counter()
mini_vectors = mini_model.encode(
    texts,
    normalize_embeddings=True
)
mini_time = time.perf_counter() - start

print("MiniLM shape:", mini_vectors.shape)
print("MiniLM time:", round(mini_time, 4))

# OpenAI embeddings
api_key = os.getenv("OPENAI_API_KEY")

if not api_key or api_key == "your_api_key_here":
    print("OpenAI comparison skipped: API key not configured.")
else:
    client = OpenAI(api_key=api_key)

    start = time.perf_counter()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    openai_vectors = np.array(
        [item.embedding for item in response.data]
    )

    openai_time = time.perf_counter() - start

    print("OpenAI shape:", openai_vectors.shape)
    print("OpenAI time:", round(openai_time, 4))

    print("OpenAI vs MiniLM comparison completed.")
