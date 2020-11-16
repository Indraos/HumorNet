import transformers
import torch

from Phyme import Phyme, rhymeUtils as ru
import itertools

import random

import matplotlib.pyplot as plt

ph = Phyme()
def get_rhymes(word):
    try:
        ns = ru.count_syllables(word)
        words = ph.get_perfect_rhymes(word)
        if ns in words:
            return words[ns]
        else:
            return list(itertools.chain.from_iterable(ph.get_perfect_rhymes(word).values()))
    except:
        return [word]


def is_letter(char):
    return (char >= 'A' and char <= 'Z') or (char >= 'a' and char <= 'z')


def swap(string,prob):
    out = ""
    last = 0
    building = False
    for index,char in enumerate(string):
        if is_letter(char) and not building:
            last = index
            building = True
        elif (not is_letter(char)) and (not building):
            out+=char
        elif (not is_letter(char)) and building:
            building = False
            if prob >= random.random():
                r = get_rhymes(string[last:index])
                new = r[int(len(r)*random.random())]
                for char2 in new:
                    if is_letter(char2):
                        out+=char2
            else:
                out += string[last:index]
            out+=char
                
    if building:
        if prob >= random.random():
            r = get_rhymes(string[last:index])
            new = r[int(len(r)*random.random())]
            for char2 in new:
                if is_letter(char2):
                    out+=char2
        else:
            out += string[last:]
        
    return out


def evaluate(text, prob, tokenizer, model):
    swap_text = swap(text,prob)
    tokenized = tokenizer(swap_text, return_tensors="pt")
    logits = model(**tokenized).logits
    results = torch.softmax(logits, dim=1).tolist()[0]
    #print(logits)
    print(swap_text)
    return results[0]


def best_prob(generator,tokenizer,model,string="",max_length=5, step = 10):
    text = generator(string, max_length)[0]['generated_text']
    x = []
    y = []
    z = []
    for p in range(step):
        x.append(p/step)
        detected = evaluate(text,p/step,tokenizer,model)
        y.append(detected)
        z.append(detected*p/step)
    plt.plot(x,y)
    plt.plot(x,z)
    plt.show()
    print(y)




# generator = transformers.pipeline("text-generation")
# tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
# model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=True)
