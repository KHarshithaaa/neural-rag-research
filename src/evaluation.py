
import re

def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).split()


def calculate_recall(expected_answer, retrieved_context):
    answer_words = set(normalize(expected_answer))
    context_words = set(normalize(retrieved_context))

    if not answer_words:
        return 0.0

    matched = answer_words.intersection(context_words)

    return len(matched) / len(answer_words)


def calculate_mrr(relevance_scores):
    for rank, relevant in enumerate(relevance_scores, start=1):
        if relevant:
            return 1 / rank
    return 0.0
