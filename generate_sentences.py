import transformers
import torch


generator = transformers.pipeline("text-generation")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", return_dict=True)


def is_letter(char):
    return (char >= 'A' and char <= 'Z') or (char >= 'a' and char <= 'z')


def create_splice(generator,tokenizer,model,string="",max_length=10):
    text = generator(string, max_length)[0]['generated_text']
    out = ""
    for c in text:
        if c!='\n':
            out += c
        else:
            out += ' '

    return out+"\n"


f = open("sentences.txt", "a")

for i in range(100):
    print(i)
    f.write(create_splice(generator,tokenizer,model))
    
f.close()


