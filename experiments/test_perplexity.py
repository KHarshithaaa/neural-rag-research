from src.perplexity import PerplexityEvaluator

evaluator = PerplexityEvaluator()

texts = [
    "Python is a programming language.",
    "RAG combines retrieval and generation."
]

for text in texts:
    print("Text:", text)
    print("Perplexity:", evaluator.calculate(text))