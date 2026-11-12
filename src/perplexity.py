import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

class PerplexityEvaluator:

    def __init__(self):
        name = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name)
        self.model.eval()

    def calculate(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                labels=inputs["input_ids"]
            )

        return math.exp(outputs.loss.item())