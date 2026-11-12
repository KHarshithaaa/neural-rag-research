from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()

rag.index_documents("data/research_dataset.jsonl")

question = "What is machine learning?"

results = rag.retrieve(question)

print("\nQuestion:", question)
print("\nRetrieved Context:")

for document in results["documents"][0]:
    print("-", document)