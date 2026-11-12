
from src.document_loader import load_documents
from src.chunking import chunk_text
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.answer_generator import AnswerGenerator


class RAGPipeline:

    def __init__(self):
        self.embedding_model = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()

    def index_documents(self, file_path):
        docs = load_documents(file_path)
        all_chunks = []

        for doc in docs:
            all_chunks.extend(
                chunk_text(doc["context"], 200, 40)
            )

        embeddings = self.embedding_model.generate_embeddings(
            all_chunks
        )

        self.vector_store.add_documents(all_chunks, embeddings)

        print("Documents indexed:", len(all_chunks))

    def ask(self, question):
        query_embedding = self.embedding_model.generate_embeddings(
            [question]
        )[0]

        results = self.vector_store.search(query_embedding, top_k=3)

        context = "\n".join(results["documents"][0])

        answer = self.answer_generator.generate(
            question, context
        )

        return {
            "question": question,
            "context": context,
            "answer": answer
        }
