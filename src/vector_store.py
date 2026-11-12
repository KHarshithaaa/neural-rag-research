import chromadb

class VectorStore:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="data/chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name="research_documents"
        )

    def add_documents(self, documents, embeddings):
        self.collection.upsert(
            ids=[str(i) for i in range(len(documents))],
            documents=documents,
            embeddings=embeddings.tolist()
        )

    def search(self, query_embedding, top_k=3):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )