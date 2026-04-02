from transformers import AutoTokenizer

class LLMUtils:

    def __init__(self, model_name):  # Constructor
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Method to build prompt from the data
    def build_prompt(self, question, context):
        if isinstance(context, list):
            context = " ".join(context)
        return (
            f"Question: {question}\n"
            f"Context: {context}\n"
            f"Answer with one word: yes, no, or maybe.\n"
            f"Answer:"
        )

    # Method to tokenize the data for BioGpt LLM
    def tokenize_data(self, row):
        max_token_len = 512

        prompt = self.build_prompt(row["question"], row["context"])
        target = f" {row["label"]}"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        # Adjusting the prompt length to have room for target
        max_prompt_len = max_token_len - len(target_ids)
        prompt_ids = prompt_ids[:max_prompt_len]

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        # Masking the prompt, so model can learn only the target
        labels = [-100] * len(prompt_ids) + target_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    # Method to normalize the output from the generated text
    def normalize_label(self, text):
        text = text.lower().strip()
        if "yes" in text:
            return "yes"
        if "no" in text:
            return "no"
        if "maybe" or "may" in text:
            return "maybe"
        return text