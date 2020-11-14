import transformers
import torch

generator = transformers.pipeline("text-generation")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=True)

text = generator("", max_length=300)[0]['generated_text']
tokenized = tokenizer(text, return_tensors="pt")
logits = model(**tokenized).logits
results = torch.softmax(logits, dim=1).tolist()[0]
