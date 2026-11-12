
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class AnswerGenerator:

    def __init__(self):
        model_name = "google/flan-t5-small"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, question, context):
        prompt = f"""
        Answer the question using the context.
        Context: {context}
        Question: {question}
        Answer:
        """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
