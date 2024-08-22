from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, question, context):
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(input_text, max_length=1024, truncation=True, return_tensors="pt")
        
        summary_ids = self.model.generate(inputs["input_ids"], num_beams=4, max_length=100, early_stopping=True)
        answer = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return answer