import json

def load_documents(file_path):
    documents = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            documents.append({
                "context": record["context"],
                "question": record["question"],
                "answer": record["answers"]["text"][0]
            })

    return documents


if __name__ == "__main__":
    docs = load_documents("data/research_dataset.jsonl")
    print("Documents loaded:", len(docs))
    print("Sample:", docs[0]["context"][:200])