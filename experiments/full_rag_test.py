
from src.rag_pipeline import RAGPipeline
from src.document_loader import load_documents

rag = RAGPipeline()

docs = load_documents("data/research_dataset.jsonl")

rag.index_documents("data/research_dataset.jsonl")

question = docs[0]["question"]

result = rag.ask(question)

print("\nQUESTION:", question)
print("\nEXPECTED ANSWER:", docs[0]["answer"])
print("\nGENERATED ANSWER:", result["answer"])
print("\nRETRIEVED CONTEXT:", result["context"][:500])
