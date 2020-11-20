import transformers
import torch

from Phyme import Phyme, rhymeUtils as ru
import itertools

import random

import matplotlib.pyplot as plt

import numpy as np

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


def evaluate(text, prob, tokenizer, model, iters=5):
    out = np.zeros((iters,iters))
    if prob == 0.0:
        print("ay")
        tokenized = tokenizer(text, return_tensors="pt")
        logits = model(**tokenized).logits
        return logits.tolist()[0][1]

    for i in range(iters):
        for j in range(iters):
            print(iters*i+j)
            swap_text = swap(text,prob)
            tokenized = tokenizer(swap_text, return_tensors="pt")
            logits = model(**tokenized).logits
            out[i][j]=logits.tolist()[0][1]
        # results = torch.softmax(logits, dim=1).tolist()[0]
    # print("HHHHHH\n",logits.tolist())
    # print("SUM:\n",logits.tolist()[0][0]+logits.tolist()[0][1])
    #print(logits)
    #print(swap_text)
    return np.median(np.mean(out,1))

def normalize(l):
    if len(l) == 0:
        return []
    minL = l[0]
    maxL = l[0]
    for i in l:
        if i<minL:
            minL=i
        if i>maxL:
            maxL=i
    out = []
    for i in l:
        out.append((i-minL)/(maxL-minL))
    return out

def best_prob(generator,tokenizer,model,string="",max_length=5, step = 10, cap = 0.5):
    text = generator(string, max_length)[0]['generated_text']
    x = []
    y = []
    z = []
    for p in range(step):
        x.append(cap*p/step)
        y.append(evaluate(text,cap*p/step,tokenizer,model))
    w = normalize(y)
    plt.plot(x,w)
    for p in range(step):
        z.append(x[p]*w[p])
    plt.plot(x,z)
    plt.show()
    print(y)




# generator = transformers.pipeline("text-generation")
# tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
# model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=True)
