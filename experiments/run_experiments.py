
import csv
import time
from experiment_config import EXPERIMENTS
from src.document_loader import load_documents
from src.chunking import chunk_text
from src.embeddings import EmbeddingGenerator

docs = load_documents("data/research_dataset.jsonl")
embedder = EmbeddingGenerator()

with open("results/experiment_results.csv", "w", newline="",
          encoding="utf-8") as file:

    writer = csv.writer(file)
    writer.writerow([
        "experiment_id", "chunk_size", "overlap",
        "top_k", "chunk_count", "embedding_time"
    ])

    for config in EXPERIMENTS:
        chunks = []

        for doc in docs:
            chunks.extend(
                chunk_text(
                    doc["context"],
                    config["chunk_size"],
                    config["overlap"]
                )
            )

        start = time.perf_counter()
        embeddings = embedder.generate_embeddings(chunks)
        elapsed = time.perf_counter() - start

        writer.writerow([
            config["id"],
            config["chunk_size"],
            config["overlap"],
            config["top_k"],
            len(chunks),
            round(elapsed, 4)
        ])

        print(
            f"Trial {config['id']}/15 completed | "
            f"Chunks: {len(chunks)} | "
            f"Embedding time: {elapsed:.2f}s"
        )

print("All 15 trials completed!")