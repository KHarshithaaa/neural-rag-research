
import json
import time
import csv

from src.document_loader import load_documents
from src.chunking import chunk_text
from src.embeddings import EmbeddingGenerator
from src.evaluation import calculate_recall, calculate_mrr

docs = load_documents("data/research_dataset.jsonl")
embedder = EmbeddingGenerator()

chunks = []

for doc in docs:
    chunks.extend(chunk_text(doc["context"], 200, 40))

document_embeddings = embedder.generate_embeddings(chunks)

results = []

for doc in docs:
    start = time.perf_counter()

    query_embedding = embedder.generate_embeddings(
        [doc["question"]]
    )[0]

    similarities = document_embeddings @ query_embedding

    top_indices = similarities.argsort()[::-1][:5]

    retrieved = [chunks[i] for i in top_indices]

    context = " ".join(retrieved)

    recall = calculate_recall(doc["answer"], context)

    relevance = [
        calculate_recall(doc["answer"], chunk) >= 0.5
        for chunk in retrieved
    ]

    mrr = calculate_mrr(relevance)

    latency = time.perf_counter() - start

    results.append({
        "recall": recall,
        "mrr": mrr,
        "latency": latency
    })

with open("results/evaluation_results.csv", "w",
          newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["recall", "mrr", "latency"]
    )

    writer.writeheader()
    writer.writerows(results)

print("Evaluation completed!")
print("Average Recall:", sum(x["recall"] for x in results) / len(results))
print("Average MRR:", sum(x["mrr"] for x in results) / len(results))
print("Average Latency:", sum(x["latency"] for x in results) / len(results))
